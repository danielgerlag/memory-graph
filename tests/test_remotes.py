import hashlib
import json
import os
import shlex
import subprocess
import uuid

import pytest
from memory_graph import git_store, remotes, schema


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def stores(tmp_path):
    upstream = tmp_path / "upstream"
    remote = tmp_path / "remote.git"
    clone = tmp_path / "clone"
    git_store.init_store(upstream)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(upstream, "remote", "add", "origin", str(remote))
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    subprocess.run(
        ["git", "clone", "--branch", "main", str(remote), str(clone)],
        check=True,
        capture_output=True,
    )
    return upstream, remote, clone


@pytest.fixture
def ssh_stores(stores, monkeypatch):
    upstream, remote, root = stores
    directory = root.parent / "ssh-bin"
    directory.mkdir()
    ssh = directory / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(directory / 'connections'))}\n"
        'case "$*" in\n'
        f"  *git-upload-pack*) exec git-upload-pack {shlex.quote(str(remote))} ;;\n"
        f"  *git-receive-pack*) exec git-receive-pack {shlex.quote(str(remote))} ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.setenv("GIT_SSH_VARIANT", "ssh")
    git(root, "remote", "set-url", "origin", "ssh://example.invalid/store.git")
    return upstream, remote, root


def commit(root, changes, ref="refs/heads/main"):
    base = git(root, "rev-parse", "refs/heads/main")
    expected = base if ref == "refs/heads/main" else "0" * len(base)
    return git_store.commit_snapshot(root, base, changes, "test change", ref, expected)


def proposal(root, operation_id):
    base = git(root, "rev-parse", "refs/heads/main")
    node_id = str(uuid.uuid4())
    timestamp = "2026-11-12T18:00:00Z"
    request = {"id": node_id, "label": "Test memory", "kind": "test", "body": "Proposal\n"}
    request_bytes = json.dumps(
        {"action": "create", "request": request},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    node = schema.dump_node(
        {
            "schema": 1,
            "id": node_id,
            "label": request["label"],
            "kind": request["kind"],
            "status": "active",
            "created_at": timestamp,
            "updated_at": timestamp,
            "aliases": [],
            "tags": [],
            "edges": [],
        },
        request["body"].encode(),
    )
    node_path = f"{node_id}.md"
    receipt = {
        "schema": 1,
        "operation_id": operation_id,
        "request_hash": hashlib.sha256(request_bytes).hexdigest(),
        "action": "create",
        "base": base,
        "preconditions": {},
        "node_ids": [node_id],
        "resolved_at": timestamp,
        "generated_ids": {},
        "changed_hashes": {node_path: hashlib.sha256(node).hexdigest()},
    }
    return commit(
        root,
        {node_path: node, f"operations/{operation_id}.json": json.dumps(receipt).encode()},
        f"refs/heads/memory-graph/{operation_id}",
    )


def test_attach_existing_snapshot_and_fetch_explicitly(stores, monkeypatch):
    upstream, _, root = stores
    assert remotes.get_config(root) is None
    config = remotes.attach(root, "origin", "propose")
    assert config["fetched_at"] is None
    store_id = json.loads((upstream / "store.json").read_text())["id"]
    assert config["store_id"] == store_id
    assert remotes.get_config(root) == config
    assert set(config) == {
        "remote",
        "mode",
        "store_id",
        "fetched_at",
        "fetched_commit",
        "last_fetch",
        "endpoint_id",
    }
    old = git(root, "rev-parse", "refs/remotes/origin/main")
    new = commit(upstream, {"notes.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    assert remotes.status(root)["observed_commit"] == old
    assert remotes.status(root)["freshness"] == "unknown"
    real_snapshot = remotes._snapshot
    verified = []

    def assert_isolated(root, ref):
        if ref.startswith("refs/memory-graph/fetch/"):
            assert git(root, "rev-parse", "refs/remotes/origin/main") == old
            verified.append(ref)
        return real_snapshot(root, ref)

    monkeypatch.setattr(remotes, "_snapshot", assert_isolated)
    fetched = remotes.fetch(root)
    assert verified
    assert fetched["fetched_at"]
    assert fetched["fetched_commit"] == new
    assert fetched["last_fetch"] == fetched["fetched_at"]
    assert fetched["store_id"] == store_id
    assert remotes.get_config(root) == fetched
    assert git(root, "rev-parse", "refs/heads/main") == old
    assert remotes.status(root)["observed_commit"] == new
    assert git(root, "for-each-ref", "--format=%(refname)", "refs/memory-graph/fetch/") == ""


def test_attach_rejects_missing_tracking_invalid_identity_or_mode(stores):
    upstream, _, root = stores
    git(root, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises(remotes.RemoteError):
        remotes.attach(root, "origin", "propose")
    git(root, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
    with pytest.raises(remotes.RemoteError):
        remotes.attach(root, "origin", "admin")
    with pytest.raises(remotes.RemoteError):
        remotes.attach(root, "--upload-pack=bad", "read")
    with pytest.raises(remotes.RemoteError):
        remotes.attach(root, "missing", "read")
    wrong = json.dumps({"schema": 1, "id": str(uuid.uuid4())}).encode()
    commit(upstream, {"store.json": wrong})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    git(root, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
    with pytest.raises(remotes.RemoteError, match="identity"):
        remotes.attach(root, "origin", "read")
    assert remotes.get_config(root) is None


def test_attach_rejects_invalid_fetched_graph_without_changing_refs(stores):
    upstream, _, root = stores
    broken = commit(upstream, {f"{uuid.uuid4()}.md": b"not valid frontmatter"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    git(root, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
    with pytest.raises(remotes.RemoteError, match="Invalid"):
        remotes.attach(root, "origin", "read")
    assert git(root, "rev-parse", "refs/remotes/origin/main") == broken
    assert remotes.get_config(root) is None


@pytest.mark.parametrize("invalid", ["graph", "identity", "manifest"])
def test_invalid_fetched_snapshot_never_updates_tracking_ref(stores, monkeypatch, invalid):
    upstream, _, root = stores
    old_config = remotes.attach(root, "origin", "read")
    old = git(root, "rev-parse", "refs/remotes/origin/main")
    if invalid == "graph":
        changes = {f"{uuid.uuid4()}.md": b"not valid frontmatter"}
    elif invalid == "identity":
        changes = {"store.json": json.dumps({"schema": 1, "id": str(uuid.uuid4())}).encode()}
    else:
        changes = {"store.json": json.dumps({"schema": 2, "id": old_config["store_id"]}).encode()}
    commit(upstream, changes)
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    real_snapshot = remotes._snapshot
    verified = []

    def assert_isolated(root, ref):
        if ref.startswith("refs/memory-graph/fetch/"):
            assert git(root, "rev-parse", "refs/remotes/origin/main") == old
            verified.append(ref)
        return real_snapshot(root, ref)

    monkeypatch.setattr(remotes, "_snapshot", assert_isolated)
    with pytest.raises(remotes.RemoteError):
        remotes.fetch(root)
    assert verified
    assert git(root, "rev-parse", "refs/remotes/origin/main") == old
    assert remotes.get_config(root) == old_config
    assert git(root, "for-each-ref", "--format=%(refname)", "refs/memory-graph/fetch/") == ""


def test_fetch_rejects_missing_memory_evidence_without_advancing_tracking_ref(stores):
    upstream, _, root = stores
    old_config = remotes.attach(root, "origin", "read")
    old = git(root, "rev-parse", "refs/remotes/origin/main")
    node_id = str(uuid.uuid4())
    node = {
        "schema": 1,
        "id": node_id,
        "label": "Derived",
        "kind": "claim",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "edges": [],
        "sources": [
            {
                "kind": "memory",
                "node_id": str(uuid.uuid4()),
                "revision": old,
                "observed_at": "2026-01-01T00:00:00Z",
            }
        ],
    }
    commit(upstream, {f"{node_id}.md": schema.dump_node(node, b"")})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    with pytest.raises(remotes.RemoteError, match="Invalid"):
        remotes.fetch(root)
    assert git(root, "rev-parse", "refs/remotes/origin/main") == old
    assert remotes.get_config(root) == old_config
    assert git(root, "for-each-ref", "--format=%(refname)", "refs/memory-graph/fetch/") == ""


def test_fetch_accepts_memory_evidence_in_fetched_history(stores):
    upstream, _, root = stores
    remotes.attach(root, "origin", "read")
    source_id = str(uuid.uuid4())
    timestamp = "2026-01-01T00:00:00Z"
    source = {
        "schema": 1,
        "id": source_id,
        "label": "Source",
        "kind": "claim",
        "status": "active",
        "created_at": timestamp,
        "updated_at": timestamp,
        "edges": [],
    }
    revision = commit(upstream, {f"{source_id}.md": schema.dump_node(source, b"")})
    node_id = str(uuid.uuid4())
    derived = {
        **source,
        "id": node_id,
        "label": "Derived",
        "sources": [
            {
                "kind": "memory",
                "node_id": source_id,
                "revision": revision,
                "observed_at": timestamp,
            }
        ],
    }
    expected = commit(upstream, {f"{node_id}.md": schema.dump_node(derived, b"")})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    assert remotes.fetch(root)["fetched_commit"] == expected


def test_fetch_can_recover_deleted_tracking_ref(stores):
    _, _, root = stores
    expected = remotes.attach(root, "origin", "read")["fetched_commit"]
    git(root, "update-ref", "-d", "refs/remotes/origin/main")
    assert remotes.fetch(root)["fetched_commit"] == expected


def test_local_file_uri_can_fetch_but_cannot_publish(stores):
    _, remote, root = stores
    git(root, "remote", "set-url", "origin", remote.as_uri())
    expected = remotes.attach(root, "origin", "propose")["fetched_commit"]
    assert remotes.fetch(root)["fetched_commit"] == expected
    with pytest.raises(remotes.RemoteError, match="Local publication is unavailable"):
        remotes.publish(root, str(uuid.uuid4()))


def test_fetch_rejects_concurrent_tracking_ref_update(stores, monkeypatch):
    upstream, _, root = stores
    old_config = remotes.attach(root, "origin", "read")
    competing = commit(root, {"notes.txt": b"other"}, "refs/heads/other")
    commit(upstream, {"notes.txt": b"remote"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    real_git = remotes._git

    def concurrent(root, *args, **kwargs):
        if args[:2] == ("update-ref", "refs/remotes/origin/main"):
            git(root, "update-ref", "refs/remotes/origin/main", competing)
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(remotes, "_git", concurrent)
    with pytest.raises(remotes.RemoteError):
        remotes.fetch(root)
    assert git(root, "rev-parse", "refs/remotes/origin/main") == competing
    assert remotes.get_config(root) == old_config
    assert git(root, "for-each-ref", "--format=%(refname)", "refs/memory-graph/fetch/") == ""


def test_fetch_config_write_failure_restores_tracking_ref(stores, monkeypatch):
    upstream, _, root = stores
    old_config = remotes.attach(root, "origin", "read")
    old = git(root, "rev-parse", "refs/remotes/origin/main")
    commit(upstream, {"notes.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")

    def fail_write(*args):
        raise OSError("write failed")

    monkeypatch.setattr(remotes, "_write_config", fail_write)
    with pytest.raises(remotes.RemoteError, match="save fetch configuration"):
        remotes.fetch(root)
    assert git(root, "rev-parse", "refs/remotes/origin/main") == old
    assert remotes.get_config(root) == old_config
    assert git(root, "for-each-ref", "--format=%(refname)", "refs/memory-graph/fetch/") == ""


def test_publication_requires_operation_receipt(ssh_stores):
    _, _, root = ssh_stores
    op = str(uuid.uuid4())
    branch = f"refs/heads/memory-graph/{op}"
    commit(root, {"notes.txt": b"not an operation"}, branch)
    remotes.attach(root, "origin", "propose")
    with pytest.raises(remotes.RemoteError, match="receipt"):
        remotes.publish(root, op)
    assert git(root, "ls-remote", "--refs", "origin", branch) == ""


def test_read_mode_blocks_publication_and_write_mode_only_proposes(ssh_stores):
    _, _, root = ssh_stores
    op = str(uuid.uuid4())
    oid = proposal(root, op)
    remotes.attach(root, "origin", "read")
    with pytest.raises(remotes.RemoteError, match="Read mode"):
        remotes.publish(root, op)
    remotes.attach(root, "origin", "write")
    original_main = git(root, "ls-remote", "--refs", "origin", "refs/heads/main").split()[0]
    result = remotes.publish(root, op)
    assert result["status"] == "proposed"
    assert result["accepted"] is False
    assert result["review_created"] is False
    assert git(root, "ls-remote", "--refs", "origin", result["ref"]).split()[0] == oid
    assert git(root, "ls-remote", "--refs", "origin", "refs/heads/main").split()[0] == original_main
    assert remotes.publish(root, op)["status"] == "reconciled"


def test_remote_conflict_is_never_overwritten(ssh_stores):
    upstream, _, root = ssh_stores
    op = str(uuid.uuid4())
    proposal(root, op)
    branch = f"refs/heads/memory-graph/{op}"
    other = commit(upstream, {"notes.txt": b"other"}, branch)
    git(upstream, "push", "origin", f"{branch}:{branch}")
    remotes.attach(root, "origin", "propose")
    with pytest.raises(remotes.RemoteError, match="conflicts"):
        remotes.publish(root, op)
    assert git(root, "ls-remote", "--refs", "origin", branch).split()[0] == other


def test_concurrent_proposal_creation_never_fast_forwards(ssh_stores, monkeypatch):
    upstream, _, root = ssh_stores
    op = str(uuid.uuid4())
    proposal(root, op)
    branch = f"refs/heads/memory-graph/{op}"
    base = git(root, "rev-parse", "refs/heads/main")
    remotes.attach(root, "origin", "propose")
    real_git = remotes._git
    inserted = []

    def concurrent(root, *args, **kwargs):
        if args[0] == "push":
            assert f"--force-with-lease={branch}:" in args
            git(upstream, "push", "origin", f"{base}:{branch}")
            inserted.append(True)
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(remotes, "_git", concurrent)
    with pytest.raises(remotes.RemoteError, match="conflicts"):
        remotes.publish(root, op)
    assert inserted
    assert git(root, "ls-remote", "--refs", "origin", branch).split()[0] == base


def test_concurrent_identical_proposal_reconciles(ssh_stores, monkeypatch):
    _, _, root = ssh_stores
    op = str(uuid.uuid4())
    oid = proposal(root, op)
    branch = f"refs/heads/memory-graph/{op}"
    remotes.attach(root, "origin", "propose")
    real_git = remotes._git

    def concurrent(root, *args, **kwargs):
        if args[0] == "push":
            git(root, "push", "--no-verify", "origin", f"{oid}:{branch}")
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(remotes, "_git", concurrent)
    assert remotes.publish(root, op)["status"] == "reconciled"
    assert git(root, "ls-remote", "--refs", "origin", branch).split()[0] == oid


@pytest.mark.parametrize("operation", ["ls-remote", "push"])
def test_timeout_reports_unknown_and_retry_reconciles(ssh_stores, monkeypatch, operation):
    _, _, root = ssh_stores
    op = str(uuid.uuid4())
    proposal(root, op)
    remotes.attach(root, "origin", "propose")
    real_git = remotes._git

    def interrupted(root, *args, **kwargs):
        if args[0] == operation:
            if operation == "push":
                real_git(root, *args, **kwargs)
            raise remotes._Timeout("timeout")
        return real_git(root, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(remotes, "_git", interrupted)
        assert remotes.publish(root, op)["status"] == "unknown"
    expected = "reconciled" if operation == "push" else "proposed"
    assert remotes.publish(root, op)["status"] == expected


def test_local_hooks_not_executed(ssh_stores):
    upstream, _, root = ssh_stores
    hook_marker = root / "hook-ran"
    hooks = root / ".git" / "hooks"
    for name in ("pre-push", "post-checkout", "post-merge"):
        path = hooks / name
        path.write_text(f"#!/bin/sh\ntouch '{hook_marker}'\n")
        path.chmod(0o755)
    remotes.attach(root, "origin", "propose")
    commit(upstream, {"notes.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    remotes.fetch(root)
    op = str(uuid.uuid4())
    proposal(root, op)
    remotes.publish(root, op)
    assert not hook_marker.exists()


def test_ext_remote_helper_does_not_execute(stores):
    _, _, root = stores
    marker = root / "ext-ran"
    helper = root / "synthetic-helper"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    remotes.attach(root, "origin", "read")
    git(root, "remote", "set-url", "origin", f"ext::{helper}")
    git(root, "config", "protocol.ext.allow", "always")
    with pytest.raises(remotes.RemoteError):
        remotes.fetch(root)
    assert not marker.exists()


def test_remote_uploadpack_command_does_not_execute(stores):
    _, _, root = stores
    marker = root / "uploadpack-ran"
    helper = root / "synthetic-uploadpack"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    git(root, "config", "remote.origin.uploadpack", str(helper))
    remotes.attach(root, "origin", "read")
    remotes.fetch(root)
    assert not marker.exists()


def test_local_alternate_refs_command_does_not_execute(stores):
    upstream, _, root = stores
    marker = root / "alternate-ran"
    helper = root / "synthetic-alternate"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 0\n")
    helper.chmod(0o755)
    (root / ".git" / "objects" / "info" / "alternates").write_text(
        str(upstream / ".git" / "objects") + "\n"
    )
    git(root, "config", "core.alternateRefsCommand", str(helper))
    new = commit(upstream, {"notes.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    git(root, "fetch", "--no-tags", "--no-write-fetch-head", "--refmap=", "origin", new)
    assert marker.exists()
    marker.unlink()
    remotes.attach(root, "origin", "read")
    assert remotes.fetch(root)["fetched_commit"] == new
    assert not marker.exists()


def test_local_upload_pack_alternate_refs_command_is_not_executed(stores):
    upstream, remote, root = stores
    marker = root / "server-alternate-ran"
    helper = root / "synthetic-server-alternate"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 0\n")
    helper.chmod(0o755)
    (remote / "objects" / "info" / "alternates").write_text(
        str(upstream / ".git" / "objects") + "\n"
    )
    subprocess.run(
        ["git", f"--git-dir={remote}", "config", "core.alternateRefsCommand", str(helper)],
        check=True,
    )
    subprocess.run(
        ["git", f"--git-dir={remote}", "rev-list", "--alternate-refs", "--all"],
        check=True,
        capture_output=True,
    )
    assert marker.exists()
    marker.unlink()
    new = commit(upstream, {"server.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    marker.unlink(missing_ok=True)
    remotes.attach(root, "origin", "read")
    missing = subprocess.run(
        ["git", "--no-lazy-fetch", "-C", str(root), "cat-file", "-e", new],
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert remotes.fetch(root)["fetched_commit"] == new
    assert not marker.exists()
    previous_config = remotes.get_config(root)
    invalid = commit(upstream, {f"{uuid.uuid4()}.md": b"invalid"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    marker.unlink(missing_ok=True)
    with pytest.raises(remotes.RemoteError, match="Invalid"):
        remotes.fetch(root)
    assert git(root, "rev-parse", "refs/remotes/origin/main") == new
    assert git(upstream, "rev-parse", "refs/heads/main") == invalid
    assert remotes.get_config(root) == previous_config
    assert not marker.exists()


def test_local_upload_pack_objects_hook_respects_protected_configuration(stores):
    upstream, remote, root = stores
    marker = root / "server-pack-hook-ran"
    hook = root / "synthetic-server-pack-hook"
    hook.write_text(f'#!/bin/sh\n: > {shlex.quote(str(marker))}\nexec "$@"\n')
    hook.chmod(0o755)
    subprocess.run(
        ["git", f"--git-dir={remote}", "config", "uploadpack.packObjectsHook", str(hook)],
        check=True,
    )
    ordinary = commit(upstream, {"ordinary.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    git(root, "fetch", "--no-tags", "--no-write-fetch-head", "--refmap=", "origin", ordinary)
    assert not marker.exists()
    control = commit(upstream, {"control.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    upload_pack = f"git -c uploadpack.packObjectsHook={shlex.quote(str(hook))} upload-pack"
    missing = subprocess.run(
        ["git", "--no-lazy-fetch", "-C", str(root), "cat-file", "-e", control],
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    git(
        root,
        "fetch",
        f"--upload-pack={upload_pack}",
        "--no-tags",
        "--no-write-fetch-head",
        "--refmap=",
        "origin",
        control,
    )
    assert marker.exists()
    marker.unlink()
    new = commit(upstream, {"application.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    marker.unlink(missing_ok=True)
    remotes.attach(root, "origin", "read")
    missing = subprocess.run(
        ["git", "--no-lazy-fetch", "-C", str(root), "cat-file", "-e", new],
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert remotes.fetch(root)["fetched_commit"] == new
    assert not marker.exists()


def test_local_git_missing_promisor_object_never_runs_helper(stores, monkeypatch):
    _, _, root = stores
    git(root, "config", "core.repositoryformatversion", "1")
    git(root, "config", "extensions.partialClone", "origin")
    git(root, "config", "remote.origin.promisor", "true")
    git(root, "config", "remote.origin.url", "https://example.invalid/store.git")
    directory = root.parent / "promisor-bin"
    directory.mkdir()
    marker = root / "promisor-helper-ran"
    helper = directory / "git-remote-https"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    monkeypatch.setenv("GIT_EXEC_PATH", str(directory))
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "https")
    missing = "a" * 40
    raw = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-p", missing], capture_output=True, check=False
    )
    assert raw.returncode != 0
    assert marker.exists()
    marker.unlink()
    with pytest.raises(remotes.RemoteError):
        remotes._git(root, "cat-file", "-p", missing)
    assert not marker.exists()


def test_status_missing_promisor_blob_never_runs_helper(stores, monkeypatch):
    _, _, root = stores
    remotes.attach(root, "origin", "read")
    blob = git(root, "rev-parse", "refs/heads/main:store.json")
    object_path = root / ".git" / "objects" / blob[:2] / blob[2:]
    assert object_path.is_file()
    object_path.unlink()
    git(root, "config", "core.repositoryformatversion", "1")
    git(root, "config", "extensions.partialClone", "promisor")
    git(root, "config", "remote.promisor.promisor", "true")
    git(root, "config", "remote.promisor.url", "https://example.invalid/store.git")
    directory = root.parent / "promisor-bin"
    directory.mkdir()
    marker = root / "promisor-helper-ran"
    helper = directory / "git-remote-https"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    monkeypatch.setenv("GIT_EXEC_PATH", str(directory))
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "https")
    raw = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-p", blob], capture_output=True, check=False
    )
    assert raw.returncode != 0
    assert marker.exists()
    marker.unlink()
    with pytest.raises(remotes.RemoteError):
        remotes.status(root)
    assert not marker.exists()


def test_local_git_requires_no_lazy_fetch_option(stores, monkeypatch):
    _, _, root = stores
    directory = root.parent / "old-git-bin"
    directory.mkdir()
    marker = root / "old-git-command-ran"
    executable = directory / "git"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--no-lazy-fetch" ]; then\n'
        '  echo "unknown option: --no-lazy-fetch" >&2\n'
        "  exit 129\n"
        "fi\n"
        f": > {shlex.quote(str(marker))}\n"
        "exit 1\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    with pytest.raises(remotes.RemoteError, match="Git must support --no-lazy-fetch"):
        remotes._git(root, "rev-parse", "--show-toplevel")
    assert not marker.exists()


def test_local_git_uses_shared_alternate_refs_guard(stores, monkeypatch):
    _, _, root = stores
    real_run = remotes.subprocess.run
    observed = []

    def inspect(command, **kwargs):
        assert f"core.alternateRefsCommand={os.devnull}" in command
        assert "core.alternateRefsCommand=true" not in command
        observed.append(True)
        return real_run(command, **kwargs)

    monkeypatch.setattr(remotes.subprocess, "run", inspect)
    assert remotes._git(root, "rev-parse", "--show-toplevel") == str(root)
    assert observed


def test_remote_receivepack_command_does_not_execute(ssh_stores):
    _, _, root = ssh_stores
    marker = root / "receivepack-ran"
    helper = root / "synthetic-receivepack"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    git(root, "config", "remote.origin.receivepack", str(helper))
    remotes.attach(root, "origin", "propose")
    op = str(uuid.uuid4())
    proposal(root, op)
    assert remotes.publish(root, op)["status"] == "proposed"
    assert not marker.exists()


def test_push_ignores_signing_program_and_follow_tags(ssh_stores):
    _, remote, root = ssh_stores
    marker = root / "signing-ran"
    signer = root / "synthetic-signer"
    signer.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    signer.chmod(0o755)
    git(root, "config", "push.gpgSign", "true")
    git(root, "config", "push.followTags", "true")
    git(root, "config", "gpg.program", str(signer))
    subprocess.run(
        ["git", f"--git-dir={remote}", "config", "receive.certNonceSeed", "synthetic"],
        check=True,
    )
    remotes.attach(root, "origin", "propose")
    op = str(uuid.uuid4())
    oid = proposal(root, op)
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "tag",
        "-a",
        "sensitive-tag",
        "-m",
        "synthetic",
        oid,
    )
    assert remotes.publish(root, op)["status"] == "proposed"
    assert not marker.exists()
    assert git(root, "ls-remote", "--tags", "origin") == ""


def test_push_destination_drift_fails_closed(ssh_stores):
    _, _, root = ssh_stores
    remotes.attach(root, "origin", "propose")
    op = str(uuid.uuid4())
    proposal(root, op)
    git(root, "remote", "set-url", "--push", "origin", "ssh://other.invalid/store.git")
    with pytest.raises(remotes.RemoteError, match="Push destination differs"):
        remotes.publish(root, op)


def test_config_changes_during_publish_cannot_redirect(ssh_stores, monkeypatch):
    _, remote, root = ssh_stores
    remotes.attach(root, "origin", "propose")
    op = str(uuid.uuid4())
    oid = proposal(root, op)
    real_git = remotes._git
    injected = []

    def concurrent(root, *args, **kwargs):
        if args[0] == "push":
            git(root, "remote", "set-url", "--push", "origin", "ssh://other.invalid/store.git")
            git(
                root,
                "config",
                "url.ssh://other.invalid/.insteadOf",
                "ssh://example.invalid/",
            )
            assert args[-2] == "ssh://example.invalid/store.git"
            injected.append(True)
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(remotes, "_git", concurrent)
    assert remotes.publish(root, op)["status"] == "proposed"
    assert injected
    result = subprocess.run(
        ["git", f"--git-dir={remote}", "rev-parse", f"refs/heads/memory-graph/{op}"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == oid
    connections = (root.parent / "ssh-bin" / "connections").read_text()
    assert "example.invalid" in connections
    assert "other.invalid" not in connections


def test_config_changes_during_fetch_cannot_redirect(stores, monkeypatch):
    upstream, _, root = stores
    remotes.attach(root, "origin", "read")
    new = commit(upstream, {"notes.txt": b"next"})
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    marker = root / "race-ran"
    helper = root / "synthetic-race"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    real_git = remotes._git
    injected = []

    def concurrent(root, *args, **kwargs):
        if args[0] == "fetch" and kwargs.get("transport"):
            git(root, "remote", "set-url", "origin", f"ext::{helper}")
            git(root, "config", "protocol.ext.allow", "always")
            injected.append(True)
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(remotes, "_git", concurrent)
    assert remotes.fetch(root)["fetched_commit"] == new
    assert injected
    assert not marker.exists()
    with pytest.raises(remotes.RemoteError):
        remotes.fetch(root)


def test_transport_configuration_tampering_fails_closed(stores):
    _, _, root = stores
    remotes.attach(root, "origin", "read")
    remotes.fetch(root)
    config = root / ".memory-graph" / "transport.git" / "config"
    config.chmod(0o600)
    config.write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = true\n[http]\n\tsslVerify = false\n"
    )
    with pytest.raises(remotes.RemoteError, match="Unsafe transport repository configuration"):
        remotes.fetch(root)


def test_sha256_local_fetch_uses_matching_object_format(tmp_path):
    upstream = tmp_path / "sha256-upstream"
    remote = tmp_path / "sha256-remote.git"
    root = tmp_path / "sha256-clone"
    subprocess.run(
        ["git", "init", "--initial-branch=main", "--object-format=sha256", str(upstream)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "init", "--bare", "--object-format=sha256", str(remote)],
        check=True,
        capture_output=True,
    )
    (upstream / "store.json").write_text(json.dumps({"schema": 1, "id": str(uuid.uuid4())}))
    git(upstream, "add", "store.json")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initialize",
    )
    git(upstream, "remote", "add", "origin", str(remote))
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    subprocess.run(
        ["git", "clone", "--branch", "main", str(remote), str(root)],
        check=True,
        capture_output=True,
    )
    remotes.attach(root, "origin", "read")
    assert remotes.fetch(root)["fetched_commit"] == git(root, "rev-parse", "refs/heads/main")


@pytest.mark.parametrize(
    "key",
    [
        "core.sshCommand",
        "core.askPass",
        "credential.helper",
        "credential.https://example.invalid.helper",
        "remote.origin.vcs",
    ],
)
def test_local_command_configuration_is_rejected(stores, key):
    _, _, root = stores
    remotes.attach(root, "origin", "read")
    git(root, "config", key, "synthetic-command")
    with pytest.raises(remotes.RemoteError):
        remotes.fetch(root)


def test_rewritten_ext_remote_is_ignored(stores):
    _, remote, root = stores
    remotes.attach(root, "origin", "read")
    marker = root / "rewrite-ran"
    helper = root / "synthetic-rewrite"
    helper.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    git(root, "config", "protocol.ext.allow", "always")
    git(root, "config", f"url.ext::{helper}.insteadOf", str(remote))
    assert remotes.fetch(root)["fetched_commit"] == git(root, "rev-parse", "refs/heads/main")
    assert not marker.exists()


def test_ssh_transport_ignores_environment_command(ssh_stores, monkeypatch):
    _, _, root = ssh_stores
    marker = root / "ssh-command-ran"
    ssh = root / "untrusted-ssh"
    ssh.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    ssh.chmod(0o755)
    monkeypatch.setenv("GIT_SSH_COMMAND", shlex.quote(str(ssh)))
    remotes.attach(root, "origin", "read")
    assert remotes.fetch(root)["fetched_commit"] == git(root, "rev-parse", "refs/heads/main")
    assert not marker.exists()


def test_https_transport_ignores_tls_proxy_and_credential_redirection(stores, monkeypatch):
    _, _, root = stores
    endpoint = "https://example.invalid/store.git"
    monkeypatch.setenv("HOME", str(root.parent))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root.parent / "xdg"))
    (root.parent / ".gitconfig").write_text("[credential]\n\thelper = synthetic-helper\n")
    git(root, "remote", "set-url", "origin", endpoint)
    git(root, "config", "http.sslVerify", "false")
    git(root, "config", "http.https://example.invalid.sslVerify", "false")
    git(root, "config", "http.followRedirects", "true")
    git(root, "config", "http.proxy", "http://attacker.invalid")
    git(root, "config", "http.extraHeader", "Authorization: synthetic")
    git(root, "config", "url.https://attacker.invalid/.insteadOf", "https://example.invalid/")
    monkeypatch.setenv("GIT_SSL_NO_VERIFY", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid")
    monkeypatch.setenv("GIT_ASKPASS", "synthetic-askpass")
    remotes.attach(root, "origin", "read")
    old = git(root, "rev-parse", "refs/heads/main")
    real_git = remotes._git
    real_run = remotes.subprocess.run
    observed = []

    def inspect_run(command, **kwargs):
        if "--git-dir" in command:
            env = kwargs["env"]
            assert "GIT_SSL_NO_VERIFY" not in env
            assert "HTTPS_PROXY" not in env
            assert "GIT_ASKPASS" not in env
            assert env["GIT_CONFIG_GLOBAL"] == os.devnull
            observed.append(True)
        return real_run(command, **kwargs)

    def simulate_transport(root, *args, **kwargs):
        if kwargs.get("transport") and args[0] == "ls-remote":
            assert args[-2] == endpoint
            credentials = kwargs["credentials"]
            assert credentials == (("credential.helper", "synthetic-helper"),)
            assert (
                real_git(
                    root,
                    "config",
                    "--get",
                    "credential.helper",
                    transport=True,
                    credentials=credentials,
                )
                == "synthetic-helper"
            )
            assert (
                real_git(
                    root, "config", "--get-urlmatch", "http.sslVerify", endpoint, transport=True
                )
                == "true"
            )
            assert (
                real_git(root, "config", "--get", "http.followRedirects", transport=True) == "false"
            )
            assert real_git(root, "config", "--get", "http.proxy", transport=True) == ""
            with pytest.raises(remotes.RemoteError):
                real_git(root, "config", "--get", "http.extraHeader", transport=True)
            return f"{old}\trefs/heads/main"
        if kwargs.get("transport") and args[0] == "fetch":
            assert args[-2] == endpoint
            return ""
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(remotes.subprocess, "run", inspect_run)
    monkeypatch.setattr(remotes, "_git", simulate_transport)
    assert remotes.fetch(root)["fetched_commit"] == old
    assert observed


def test_local_remote_server_hooks_not_executed(stores):
    _, remote, root = stores
    marker = root / "server-hook-ran"
    hook = remote / "hooks" / "pre-receive"
    hook.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 0\n")
    hook.chmod(0o755)
    remotes.attach(root, "origin", "propose")
    op = str(uuid.uuid4())
    proposal(root, op)
    with pytest.raises(remotes.RemoteError, match="Local publication is unavailable"):
        remotes.publish(root, op)
    assert not marker.exists()


def test_ssh_server_policy_hook_is_not_bypassed(ssh_stores):
    _, remote, root = ssh_stores
    marker = root / "server-policy-ran"
    hook = remote / "hooks" / "pre-receive"
    hook.write_text(f"#!/bin/sh\n: > {shlex.quote(str(marker))}\nexit 1\n")
    hook.chmod(0o755)
    remotes.attach(root, "origin", "propose")
    op = str(uuid.uuid4())
    proposal(root, op)
    assert remotes.publish(root, op)["status"] == "unknown"
    assert marker.exists()
    assert git(root, "ls-remote", "--refs", "origin", f"refs/heads/memory-graph/{op}") == ""


def test_errors_never_expose_remote_url(stores):
    _, _, root = stores
    secret_url = "https://user:secret@example.invalid/repository"
    git(root, "remote", "set-url", "origin", secret_url)
    with pytest.raises(remotes.RemoteError) as error:
        remotes.attach(root, "origin", "propose")
    assert "secret" not in str(error.value)
    assert remotes.get_config(root) is None


def test_bad_configuration_fails_closed(stores):
    _, _, root = stores
    remotes.attach(root, "origin", "read")
    config = root / ".memory-graph" / "config.json"
    config.write_text('{"remote": "origin", "remote": "other"}')
    with pytest.raises(remotes.RemoteError, match="configuration"):
        remotes.get_config(root)
    assert not list(config.parent.glob(".config-*.tmp"))
    assert os.stat(config).st_mode & 0o077 == 0


def test_legacy_attachment_requires_explicit_reattach(stores):
    _, _, root = stores
    config = remotes.attach(root, "origin", "read")
    del config["endpoint_id"]
    (root / ".memory-graph" / "config.json").write_text(json.dumps(config))
    with pytest.raises(remotes.RemoteError, match="explicitly reattach"):
        remotes.get_config(root)
