import json
import os
import shlex
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from memory_graph import git_store
from memory_graph.git_store import (
    GitStoreError,
    RefNotFound,
    StaleRefError,
    commit_snapshot,
    init_store,
    load_snapshot,
)


def git(root, *args, data=None, check=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
    )
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=data,
        capture_output=True,
        check=check,
        env=env,
        timeout=30,
    ).stdout


def promisor_marker(root, tmp_path):
    marker = tmp_path / "remote-helper-ran"
    helper = tmp_path / "fake-upload-pack"
    helper.write_text(f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    for key, value in (
        ("core.repositoryformatversion", "1"),
        ("extensions.partialClone", "danger"),
        ("remote.danger.promisor", "true"),
        ("remote.danger.url", "ext::" + str(helper).replace("%", "%%").replace(" ", "% ")),
        ("protocol.ext.allow", "always"),
    ):
        git(root, "config", key, value)
    return marker


def unprotected_control(root, args, data=None):
    env = git_store._environment()
    env.pop("GIT_ALLOW_PROTOCOL")
    env.pop("GIT_NO_LAZY_FETCH")
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=data,
        capture_output=True,
        env=env,
        check=False,
        timeout=30,
    )


def test_init_manifest_history_and_preserves_unrelated_files(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    (root / "untracked").write_bytes(b"untouched")
    commit = init_store(root)
    resolved, snapshot = load_snapshot(root)
    manifest = json.loads(snapshot["store.json"])
    assert manifest["schema"] == 1
    assert uuid.UUID(manifest["id"]).version == 4
    assert snapshot[".gitignore"] == b".memory-graph/\n"
    assert resolved == commit
    assert set(snapshot) == {"store.json", ".gitignore"}
    assert (root / "untracked").read_bytes() == b"untouched"
    assert git(root, "symbolic-ref", "HEAD").strip() == b"refs/heads/main"
    assert git(root, "rev-list", "--count", "main").strip() == b"1"
    assert git(root, "status", "--porcelain").strip() == b"?? untracked"


@pytest.mark.parametrize("name", [".git", ".gitignore", "store.json"])
def test_init_refuses_existing_files_and_repositories(tmp_path, name):
    root = tmp_path / "store"
    root.mkdir()
    (root / name).write_bytes(b"existing")
    with pytest.raises(GitStoreError):
        init_store(root)
    assert (root / name).read_bytes() == b"existing"
    assert not (root / ".git").is_dir()


def test_existing_repo_and_plain_or_subdirectory_are_rejected(tmp_path):
    root = tmp_path / "store"
    init_store(root)
    with pytest.raises(GitStoreError):
        init_store(root)
    subdir = root / "subdir"
    subdir.mkdir()
    for path in (subdir, tmp_path):
        with pytest.raises(GitStoreError):
            load_snapshot(path)


def test_commits_are_atomic_snapshots_without_touching_user_work(tmp_path):
    root = tmp_path / "store"
    initial = init_store(root)
    (root / "store.json").write_bytes(b"staged")
    git(root, "add", "store.json")
    (root / "store.json").write_bytes(b"dirty")
    (root / "untracked").write_bytes(b"user")
    before_index = (root / ".git/index").read_bytes()
    second = commit_snapshot(
        root, initial, {"a/one.md": b"one\0\n", "a/two.md": b"two"}, "memory: create"
    )
    commit, snapshot = load_snapshot(root)
    assert commit == second
    assert snapshot["a/one.md"] == b"one\0\n"
    assert snapshot["a/two.md"] == b"two"
    assert json.loads(snapshot["store.json"])["schema"] == 1
    assert (root / "store.json").read_bytes() == b"dirty"
    assert (root / "untracked").read_bytes() == b"user"
    assert not (root / "a").exists()
    assert (root / ".git/index").read_bytes() == before_index
    assert git(root, "rev-parse", f"{second}^").strip().decode() == initial
    assert load_snapshot(root, initial)[1].keys() == {".gitignore", "store.json"}
    third = commit_snapshot(root, second, {"a/one.md": None, "a/two.md": b"new"}, "update")
    assert load_snapshot(root)[1]["a/two.md"] == b"new"
    assert "a/one.md" not in load_snapshot(root)[1]
    assert git(root, "rev-parse", f"{third}^").strip().decode() == second


def test_stale_compare_and_swap_and_proposal_creation(tmp_path):
    root = tmp_path / "store"
    base = init_store(root)
    main = commit_snapshot(root, base, {"main": b"m"}, "main")
    with pytest.raises(StaleRefError):
        commit_snapshot(root, base, {"lost": b"x"}, "stale")
    assert load_snapshot(root)[0] == main
    proposal = commit_snapshot(
        root, base, {"proposal": b"p"}, "proposal", "refs/heads/proposals/test", "0" * len(base)
    )
    assert load_snapshot(root)[0] == main
    assert load_snapshot(root, "refs/heads/proposals/test")[1]["proposal"] == b"p"
    assert git(root, "rev-parse", f"{proposal}^").strip().decode() == base
    with pytest.raises(StaleRefError):
        commit_snapshot(
            root, base, {"other": b"x"}, "conflict", "refs/heads/proposals/test", "0" * len(base)
        )
    assert load_snapshot(root, "refs/heads/proposals/test")[0] == proposal


def test_directory_replacements_only_apply_explicit_changes(tmp_path):
    root = tmp_path / "store"
    base = init_store(root)
    base = commit_snapshot(root, base, {"a/one": b"one", "a/two": b"two"}, "create")
    base = commit_snapshot(root, base, {"a/one": None, "a/two": None, "a": b"file"}, "flatten")
    assert load_snapshot(root)[1]["a"] == b"file"
    base = commit_snapshot(root, base, {"a": None, "a/child": b"child"}, "expand")
    assert load_snapshot(root)[1]["a/child"] == b"child"


def test_read_is_pinned_before_concurrent_ref_change(tmp_path, monkeypatch):
    root = tmp_path / "store"
    base = init_store(root)
    proposal = commit_snapshot(
        root, base, {"new": b"new"}, "new", "refs/heads/proposal", "0" * len(base)
    )
    run = git_store._bounded_output
    reads = []

    def advance(root, args, max_bytes, **kwargs):
        if args[:2] == ["cat-file", "--batch-check"]:
            reads.append(args)
            git(root, "update-ref", "refs/heads/main", proposal, base)
        return run(root, args, max_bytes, **kwargs)

    monkeypatch.setattr(git_store, "_bounded_output", advance)
    snapshot_id, snapshot = load_snapshot(root)
    assert snapshot_id == base
    assert "new" not in snapshot
    assert len(reads) == 1
    assert git(root, "rev-parse", "refs/heads/main").strip().decode() == proposal


def repeated_blob_commit(root, data, count):
    oid = git(root, "hash-object", "-w", "--stdin", data=data).strip().decode()
    tree_data = b"".join(
        f"100644 blob {oid}\tfile-{number:04}.md\0".encode() for number in range(count)
    )
    tree = git(root, "mktree", "-z", data=tree_data).strip().decode()
    return git(root, "commit-tree", tree, data=b"fixture").strip().decode()


def test_blob_and_total_limits_are_checked_before_content_reads(tmp_path, monkeypatch):
    root = tmp_path / "store"
    init_store(root)
    blob = b"x" * (8 * 1024 * 1024)
    at_limit = repeated_blob_commit(root, blob, 64)
    over_total = repeated_blob_commit(root, blob, 65)
    over_blob = repeated_blob_commit(root, blob + b"x", 1)
    run = git_store._bounded_output
    calls = []

    def track(root, args, max_bytes, **kwargs):
        calls.append(args[:2])
        return run(root, args, max_bytes, **kwargs)

    monkeypatch.setattr(git_store, "_bounded_output", track)
    _, snapshot = load_snapshot(root, at_limit)
    assert len(snapshot) == 64
    assert snapshot["file-0000.md"] == blob
    assert calls.count(["cat-file", "--batch-check"]) == 1
    assert calls.count(["cat-file", "--batch"]) == 1
    for commit, error in ((over_total, "512 MiB"), (over_blob, "8 MiB")):
        calls.clear()
        with pytest.raises(GitStoreError, match=error):
            load_snapshot(root, commit)
        assert ["cat-file", "--batch-check"] in calls
        assert ["cat-file", "--batch"] not in calls


def test_snapshot_metadata_is_batched(tmp_path, monkeypatch):
    root = tmp_path / "store"
    base = init_store(root)
    commit_snapshot(root, base, {f"node-{n}": str(n).encode() for n in range(12)}, "many")
    monkeypatch.setattr(git_store, "_METADATA_BATCH_SIZE", 4)
    run = git_store._bounded_output
    calls = []

    def track(root, args, max_bytes, **kwargs):
        calls.append((args[:2], max_bytes))
        return run(root, args, max_bytes, **kwargs)

    monkeypatch.setattr(git_store, "_bounded_output", track)
    _, snapshot = load_snapshot(root)
    metadata_calls = [limit for args, limit in calls if args == ["cat-file", "--batch-check"]]
    assert len(metadata_calls) == 4
    assert all(limit <= 4 * (40 + 128) for limit in metadata_calls)
    assert len(snapshot) == 14
    assert sum(args == ["cat-file", "--batch"] for args, _ in calls) == 1


def test_blob_content_size_must_match_preflight_metadata(tmp_path, monkeypatch):
    first, second = "a" * 40, "b" * 40

    def inconsistent_output(root, args, max_bytes, **kwargs):
        if args[1] == "--batch-check":
            return f"{first} blob 2\n{second} blob 1\n".encode()
        return f"{first} blob 3\naaa\n{second} blob 0\n\n".encode()

    monkeypatch.setattr(git_store, "_bounded_output", inconsistent_output)
    with pytest.raises(GitStoreError, match="size changed after metadata check"):
        git_store._read_blobs(tmp_path, {"a": ("100644", first), "b": ("100644", second)})


@pytest.mark.skipif(os.name == "nt", reason="executable marker requires POSIX")
@pytest.mark.parametrize("during_content_read", [False, True])
def test_missing_blob_never_lazily_executes_remote_helper(
    tmp_path, monkeypatch, during_content_read
):
    root = tmp_path / "store"
    base = init_store(root)
    content = b"promisor regression blob"
    commit = commit_snapshot(root, base, {"node.md": content}, "node")
    oid = git(root, "rev-parse", f"{commit}:node.md").strip().decode()
    obj = root / ".git" / "objects" / oid[:2] / oid[2:]
    obj.unlink()
    marker = promisor_marker(root, tmp_path)
    unprotected_control(root, ["cat-file", "--batch-check"], data=f"{oid}\n".encode())
    assert marker.exists(), "positive control must trigger Git's lazy fetch"
    marker.unlink()
    deny_protocols = git_store._environment()
    deny_protocols["GIT_NO_LAZY_FETCH"] = "0"
    subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch-check"],
        input=f"{oid}\n".encode(),
        capture_output=True,
        env=deny_protocols,
        check=False,
        timeout=30,
    )
    assert not marker.exists()
    if during_content_read:
        git(root, "hash-object", "-w", "--stdin", data=content)
        run = git_store._bounded_output

        def remove_after_metadata(root, args, max_bytes, **kwargs):
            if args[:2] == ["cat-file", "--batch"]:
                obj.unlink()
            return run(root, args, max_bytes, **kwargs)

        monkeypatch.setattr(git_store, "_bounded_output", remove_after_metadata)
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "ext")
    with pytest.raises(GitStoreError, match="missing|Invalid"):
        load_snapshot(root, commit)
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="executable marker requires POSIX")
def test_missing_tree_never_lazily_executes_remote_helper(tmp_path, monkeypatch):
    root = tmp_path / "store"
    commit = init_store(root)
    tree = git(root, "rev-parse", f"{commit}^{{tree}}").strip().decode()
    (root / ".git" / "objects" / tree[:2] / tree[2:]).unlink()
    marker = promisor_marker(root, tmp_path)
    unprotected_control(root, ["ls-tree", "-r", commit])
    assert marker.exists(), "positive control must trigger Git's lazy fetch"
    marker.unlink()
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "ext")
    with pytest.raises(GitStoreError):
        load_snapshot(root)
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="executable marker requires POSIX")
def test_missing_commit_never_lazily_executes_remote_helper(tmp_path, monkeypatch):
    root = tmp_path / "store"
    commit = init_store(root)
    (root / ".git" / "objects" / commit[:2] / commit[2:]).unlink()
    marker = promisor_marker(root, tmp_path)
    unprotected_control(root, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    assert marker.exists(), "positive control must trigger Git's lazy fetch"
    marker.unlink()
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "ext")
    with pytest.raises(GitStoreError):
        load_snapshot(root, commit)
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="executable marker requires POSIX")
def test_repository_alternate_refs_command_is_never_executed(tmp_path):
    root = tmp_path / "store"
    alternate = tmp_path / "alternate"
    init_store(root)
    init_store(alternate)
    (root / ".git/objects/info/alternates").write_text(
        str(alternate / ".git/objects") + "\n", encoding="utf-8"
    )
    marker = tmp_path / "alternate-command-ran"
    helper = tmp_path / "fake-alternate-command"
    helper.write_text(f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\nexit 0\n")
    helper.chmod(0o755)
    git(root, "config", "core.alternateRefsCommand", shlex.quote(str(helper)))
    git(root, "rev-list", "--all", "--alternate-refs", check=False)
    assert marker.exists(), "positive control must execute the repository-configured command"
    marker.unlink()
    git_store._run(root, ["rev-list", "--all", "--alternate-refs"], check=False)
    assert not marker.exists()
    load_snapshot(root)
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="fake Git executable requires POSIX")
def test_unsupported_git_fails_before_initialization_or_git_subcommand(tmp_path, monkeypatch):
    real_store = tmp_path / "existing"
    init_store(real_store)
    marker = tmp_path / "subcommand-ran"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_git = fakebin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--no-lazy-fetch" ]; then\n'
        '  printf "unknown option: --no-lazy-fetch\\n" >&2\n'
        "  exit 129\n"
        "fi\n"
        f"printf ran > {shlex.quote(str(marker))}\n"
        "exit 1\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fakebin) + os.pathsep + os.environ.get("PATH", ""))
    new_store = tmp_path / "new"
    with pytest.raises(GitStoreError, match="Git 2.45 or newer"):
        init_store(new_store)
    assert not new_store.exists()
    with pytest.raises(GitStoreError, match="Git 2.45 or newer"):
        load_snapshot(real_store)
    assert not marker.exists()


def test_blob_batch_output_cannot_exceed_preflight_bound(tmp_path, monkeypatch):
    root = tmp_path / "store"
    init_store(root)
    run = git_store._bounded_output

    def smaller_content_bound(root, args, max_bytes, **kwargs):
        if args[:2] == ["cat-file", "--batch"]:
            max_bytes -= 1
        return run(root, args, max_bytes, **kwargs)

    monkeypatch.setattr(git_store, "_bounded_output", smaller_content_bound)
    with pytest.raises(GitStoreError, match="cat-file output exceeds"):
        load_snapshot(root)


def test_tree_entry_and_listing_output_limits(tmp_path, monkeypatch):
    root = tmp_path / "store"
    base = init_store(root)
    commit_snapshot(root, base, {"dir/file": b"data"}, "add")
    monkeypatch.setattr(git_store, "_MAX_TREE_ENTRIES", 4)
    assert load_snapshot(root)[1]["dir/file"] == b"data"
    monkeypatch.setattr(git_store, "_MAX_TREE_ENTRIES", 3)
    with pytest.raises(GitStoreError, match="3 entry limit"):
        load_snapshot(root)
    monkeypatch.setattr(git_store, "_MAX_TREE_ENTRIES", 4)
    monkeypatch.setattr(git_store, "_MAX_TREE_LISTING_BYTES", 8)
    with pytest.raises(GitStoreError, match="output exceeds the 8-byte limit"):
        load_snapshot(root)


def test_history_accepts_exact_commit_ids_but_not_revision_expressions(tmp_path):
    root = tmp_path / "store"
    initial = init_store(root)
    latest = commit_snapshot(root, initial, {"new": b"data"}, "next")
    assert load_snapshot(root, initial)[0] == initial
    assert load_snapshot(root, latest.upper())[0] == latest
    for expression in ("HEAD", "main", "refs/heads/main~1", f"{latest}^", latest[:12]):
        with pytest.raises(GitStoreError):
            load_snapshot(root, expression)
    git(root, "tag", "-a", "historical", "-m", "history", initial)
    tag_id = git(root, "rev-parse", "refs/tags/historical").strip().decode()
    assert load_snapshot(root, "refs/tags/historical")[0] == initial
    with pytest.raises(GitStoreError, match="must identify a commit"):
        load_snapshot(root, tag_id)


def test_simultaneous_writers_cannot_both_publish(tmp_path, monkeypatch):
    root = tmp_path / "store"
    base = init_store(root)
    barrier = threading.Barrier(2, timeout=30)
    publish = git_store._publish

    def synchronized_publish(*args):
        barrier.wait()
        return publish(*args)

    monkeypatch.setattr(git_store, "_publish", synchronized_publish)

    def writer(name):
        try:
            return commit_snapshot(root, base, {name: name.encode()}, name)
        except GitStoreError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, ["first", "second"]))
    assert sum(result is not None for result in results) == 1
    commit, snapshot = load_snapshot(root)
    assert commit in results
    assert ("first" in snapshot) != ("second" in snapshot)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute",
        "../escape",
        "a/../b",
        "a//b",
        "./x",
        ".git/config",
        ".GIT./config",
        "x\\y",
        "c:/file",
        ".memory-graph/secret",
        "a\nb",
    ],
)
def test_rejects_unsafe_paths(tmp_path, path):
    root = tmp_path / "store"
    base = init_store(root)
    with pytest.raises(GitStoreError):
        commit_snapshot(root, base, {path: b"x"}, "test")
    assert load_snapshot(root)[0] == base


def test_rejects_bad_arguments_and_tree_collisions(tmp_path):
    root = tmp_path / "store"
    base = init_store(root)
    for changes, message, ref in [
        ({"x": "not bytes"}, "test", "refs/heads/main"),
        ({"x": b"ok"}, "", "refs/heads/main"),
        ({"x": b"ok"}, "test", "--bad"),
        ({"a": b"x", "a/b": b"y"}, "test", "refs/heads/main"),
    ]:
        with pytest.raises(GitStoreError):
            commit_snapshot(root, base, changes, message, ref)
    with pytest.raises(GitStoreError):
        commit_snapshot(root, "main", {}, "test")
    with pytest.raises(RefNotFound):
        load_snapshot(root, "refs/heads/missing")
    assert load_snapshot(root)[0] == base


def test_missing_ref_is_distinct_from_broken_ref_and_noncommit(tmp_path):
    root = tmp_path / "store"
    init_store(root)
    with pytest.raises(RefNotFound):
        load_snapshot(root, "refs/heads/absent")
    (root / ".git/refs/heads/broken").write_text("not a ref\n", encoding="utf-8")
    with pytest.raises(GitStoreError):
        load_snapshot(root, "refs/heads/broken")
    blob = git(root, "hash-object", "-w", "--stdin", data=b"value").strip().decode()
    git(root, "update-ref", "refs/tags/blob", blob)
    with pytest.raises(GitStoreError):
        load_snapshot(root, "refs/tags/blob")


def test_missing_ref_fails_closed_if_git_lacks_show_ref_exists(tmp_path, monkeypatch):
    root = tmp_path / "store"
    init_store(root)
    run = git_store._run
    calls = []

    def older_git(root, args, **kwargs):
        calls.append(args[0])
        if args[:2] == ["show-ref", "--exists"]:
            return subprocess.CompletedProcess(args, 129, b"", b"unknown option")
        return run(root, args, **kwargs)

    monkeypatch.setattr(git_store, "_run", older_git)
    with pytest.raises(GitStoreError, match="show-ref --exists"):
        load_snapshot(root, "refs/heads/absent")
    (root / ".git/refs/heads/broken").write_text("not a ref\n", encoding="utf-8")
    with pytest.raises(GitStoreError, match="show-ref --exists"):
        load_snapshot(root, "refs/heads/broken")
    assert "fsck" not in calls


def test_symlinks_and_gitlinks_in_tree_are_rejected(tmp_path):
    root = tmp_path / "store"
    base = init_store(root)
    for mode in ("120000", "160000"):
        blob = git(root, "hash-object", "-w", "--stdin", data=b"target").strip().decode()
        oid = base if mode == "160000" else blob
        git(root, "update-index", "--add", "--cacheinfo", f"{mode},{oid},bad")
        tree = git(root, "write-tree").strip().decode()
        commit = git(root, "commit-tree", tree, "-p", base, data=b"bad").strip().decode()
        with pytest.raises(GitStoreError, match="non-regular"):
            load_snapshot(root, commit)
        with pytest.raises(GitStoreError, match="non-regular"):
            commit_snapshot(root, commit, {}, "reject", expected_ref=base)
        git(root, "read-tree", base)


def test_unsafe_path_in_committed_tree_is_rejected(tmp_path):
    root = tmp_path / "store"
    base = init_store(root)
    blob = git(root, "hash-object", "-w", "--stdin", data=b"value").strip().decode()
    git(root, "update-index", "--add", "--cacheinfo", f"100644,{blob},bad\nname")
    tree = git(root, "write-tree").strip().decode()
    commit = git(root, "commit-tree", tree, "-p", base, data=b"bad").strip().decode()
    with pytest.raises(GitStoreError, match="Unsafe snapshot path"):
        load_snapshot(root, commit)


def test_sha256_object_format_when_supported(tmp_path):
    root = tmp_path / "sha256"
    root.mkdir()
    result = subprocess.run(
        ["git", "-C", str(root), "init", "--object-format=sha256", "--initial-branch=main"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        pytest.skip("Git does not support SHA-256 repositories")
    tree = git(root, "mktree", data=b"").strip().decode()
    base = git(root, "commit-tree", tree, data=b"initial").strip().decode()
    git(root, "update-ref", "refs/heads/main", base)
    commit = commit_snapshot(root, base, {"a": b"data"}, "sha256")
    assert len(commit) == 64
    assert load_snapshot(root) == (commit, {"a": b"data"})


def test_git_hooks_and_filters_never_execute(tmp_path):
    root = tmp_path / "store"
    base = init_store(root)
    marker = root / "ran"
    (root / ".git/hooks").mkdir(exist_ok=True)
    for hook in ("pre-commit", "reference-transaction"):
        script = root / ".git/hooks" / hook
        script.write_text(f"#!/bin/sh\nprintf ran > '{marker}'\n", encoding="utf-8")
        script.chmod(0o755)
    git(root, "config", "filter.evil.clean", f"touch '{marker}'; cat")
    (root / ".gitattributes").write_text("*.md filter=evil\n", encoding="utf-8")
    commit = commit_snapshot(root, base, {"note.md": b"raw"}, "test")
    assert load_snapshot(root, commit)[1]["note.md"] == b"raw"
    assert not marker.exists()
    assert not list((root / ".git").glob("memory-graph-index-*"))
