"""Disposable parse cache keyed by immutable content and parser implementation."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import sqlite3
import stat
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Self

from . import schema

PARALLEL_THRESHOLD = 2000


class CacheError(RuntimeError):
    """A disposable cache is damaged and must not be trusted."""


def _parse_worker(item: tuple[str, bytes]) -> tuple[str, bytes, dict, bytes] | None:
    path, raw = item
    try:
        node, body = schema.parse_node(raw, path)
    except (ValueError, KeyError):
        return None
    return path, raw, node, body


class ParseCache:
    def __init__(self, root: Path):
        state = root / ".memory-graph"
        if state.is_symlink():
            raise ValueError("store state directory must not be a symlink")
        state.mkdir(mode=0o700, exist_ok=True)
        path = state / "parse-cache.sqlite3"
        if path.is_symlink():
            raise ValueError("parse cache must not be a symlink")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise CacheError("parse cache must be a regular file")
        if os.name != "nt" and (metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077):
            raise CacheError("parse cache is not private to its owner; remove it and retry")
        schema_file = Path(schema.__file__).resolve()
        asset = schema_file.parents[2] / "assets" / "node.schema.json"
        digest = hashlib.sha256()
        for part in (
            schema_file.read_bytes(),
            asset.read_bytes(),
            version("ruamel.yaml").encode(),
            version("jsonschema").encode(),
            str(sys.version_info[:2]).encode(),
        ):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        self.parser_id = digest.hexdigest()
        receipt_digest = hashlib.sha256()
        for part in (
            schema_file.with_name("service.py").read_bytes(),
            (asset.parent / "operation.schema.json").read_bytes(),
            version("jsonschema").encode(),
            str(sys.version_info[:2]).encode(),
        ):
            receipt_digest.update(len(part).to_bytes(8, "big"))
            receipt_digest.update(part)
        self.receipt_parser_id = receipt_digest.hexdigest()
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS nodes (parser TEXT NOT NULL, path TEXT NOT NULL, "
            "raw_hash TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL, "
            "PRIMARY KEY (parser, path, raw_hash))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS receipts (parser TEXT NOT NULL, path TEXT NOT NULL, "
            "raw_hash TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL, "
            "PRIMARY KEY (parser, path, raw_hash))"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()

    def prime(self, entries: list[tuple[str, bytes]]) -> None:
        if len(entries) < PARALLEL_THRESHOLD:
            return
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if (
            not main_file
            or str(main_file).startswith("<")
            or multiprocessing.current_process().daemon
        ):
            return
        existing = set(
            self.connection.execute(
                "SELECT path, raw_hash FROM nodes WHERE parser=?", (self.parser_id,)
            )
        )
        missing = [
            (path, raw)
            for path, raw in entries
            if (path, hashlib.sha256(raw).hexdigest()) not in existing
        ]
        if len(missing) < PARALLEL_THRESHOLD:
            return
        if sum(len(raw) for _, raw in missing) > 64 * 1024 * 1024:
            return
        workers = min(4, os.cpu_count() or 1)
        if workers == 1:
            return
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_parse_worker, missing, chunksize=32):
                if result is not None:
                    path, raw, node, body = result
                    self._store(path, hashlib.sha256(raw).hexdigest(), node, body)

    def _store(self, path: str, raw_hash: str, node: dict, body: bytes) -> None:
        payload = json.dumps(
            {"node": node, "body": base64.b64encode(body).decode("ascii")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO nodes (parser,path,raw_hash,payload,payload_hash) VALUES (?,?,?,?,?)",
            (self.parser_id, path, raw_hash, payload, hashlib.sha256(payload.encode()).hexdigest()),
        )

    def parse(self, raw: bytes, path: str) -> tuple[dict, bytes]:
        raw_hash = hashlib.sha256(raw).hexdigest()
        row = self.connection.execute(
            "SELECT payload, payload_hash FROM nodes WHERE parser=? AND path=? AND raw_hash=?",
            (self.parser_id, path, raw_hash),
        ).fetchone()
        if row is not None:
            payload, expected = row
            if hashlib.sha256(payload.encode()).hexdigest() != expected:
                raise CacheError(
                    "parse cache checksum mismatch; remove the disposable cache and retry"
                )
            try:
                decoded = json.loads(payload)
                return decoded["node"], base64.b64decode(decoded["body"], validate=True)
            except (ValueError, KeyError, TypeError) as exc:
                raise CacheError(
                    "invalid parse cache entry; remove the disposable cache and retry"
                ) from exc
        node, body = schema.parse_node(raw, path)
        self._store(path, raw_hash, node, body)
        return node, body

    def parse_receipt(self, raw: bytes, path: str, validator: Callable[[bytes, str], dict]) -> dict:
        raw_hash = hashlib.sha256(raw).hexdigest()
        row = self.connection.execute(
            "SELECT payload, payload_hash FROM receipts WHERE parser=? AND path=? AND raw_hash=?",
            (self.receipt_parser_id, path, raw_hash),
        ).fetchone()
        if row is not None:
            payload, expected = row
            if hashlib.sha256(payload.encode()).hexdigest() != expected:
                raise CacheError(
                    "receipt cache checksum mismatch; remove the disposable cache and retry"
                )
            try:
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise TypeError("receipt cache entry is not an object")
                return value
            except (ValueError, TypeError) as exc:
                raise CacheError(
                    "invalid receipt cache entry; remove the disposable cache and retry"
                ) from exc
        receipt = validator(raw, path)
        payload = json.dumps(
            receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO receipts (parser,path,raw_hash,payload,payload_hash) VALUES (?,?,?,?,?)",
            (
                self.receipt_parser_id,
                path,
                raw_hash,
                payload,
                hashlib.sha256(payload.encode()).hexdigest(),
            ),
        )
        return receipt
