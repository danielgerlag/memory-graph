import json
import subprocess
import uuid

import pytest
from memory_graph import cli, git_store, registry, remotes, service


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("MEMORY_GRAPH_DATA_DIR", str(tmp_path / "data"))
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    monkeypatch.chdir(project)
    return project


def run(*argv):
    return cli.execute(cli.parser().parse_args(argv))


def create(tmp_path, *store, label, scope="s"):
    request = tmp_path / f"{uuid.uuid4()}.json"
    request.write_text(json.dumps({"label": label, "kind": "claim", "body": "x", "scope": scope}))
    return run(*store, "create", "--operation", str(uuid.uuid4()), "--input", str(request))


def labels(*store, scope="s"):
    result = run(*store, "find", "--kind", "claim", "--scope", scope)
    return [row["label"] for row in result["results"]]


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def shared(tmp_path):
    upstream, remote = tmp_path / "upstream", tmp_path / "shared.git"
    git_store.init_store(upstream)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(upstream, "remote", "add", "origin", str(remote))
    create(tmp_path, "--root", str(upstream), label="Team fact")
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")
    return upstream, remote


def push(tmp_path, upstream, label):
    create(tmp_path, "--root", str(upstream), label=label)
    git(upstream, "push", "origin", "refs/heads/main:refs/heads/main")


def test_folder_store_is_selected_by_name_and_default(tmp_path):
    run("stores", "add", "mine", "--path", str(tmp_path / "mine"), "--init")
    create(tmp_path, "--store", "mine", label="Kept")
    assert labels("--store", "mine") == ["Kept"]
    assert labels() == ["Kept"]  # the first store added becomes the global default
    [row] = run("stores", "list")["stores"]
    assert (row["name"], row["scope"], row["state"], row["default"]) == (
        "mine",
        "global",
        "ready",
        True,
    )


def test_project_store_shadows_global_and_needs_a_default(tmp_path, isolated):
    run("stores", "add", "notes", "--path", str(tmp_path / "global"), "--init")
    run("stores", "add", "notes", "--path", str(isolated / "local"), "--init", "--scope", "project")
    create(tmp_path, "--store", "notes", label="Project only")
    assert labels("--store", "notes") == ["Project only"]
    rows = run("stores", "list")["stores"]
    assert [(r["scope"], r["shadowed"]) for r in rows] == [("project", False), ("global", True)]
    run("stores", "add", "other", "--path", str(tmp_path / "other"), "--init")
    run("stores", "remove", "notes", "--scope", "project")
    run("stores", "remove", "notes")
    with pytest.raises(registry.RegistryError, match=r"no store selected.*other"):
        labels()
    assert (isolated / registry.PROJECT_FILE).is_file()


def test_remote_store_is_cloned_read_only_and_synced(tmp_path, shared):
    upstream, remote = shared
    added = run("stores", "add", "team", "--remote", str(remote), "--mode", "read")
    assert added["mode"] == "read"
    assert labels("--store", "team") == ["Team fact"]
    with pytest.raises(service.GraphError, match="read-only"):
        create(tmp_path, "--store", "team", label="Refused")
    push(tmp_path, upstream, "Second fact")
    assert labels("--store", "team") == ["Team fact"]
    assert run("stores", "sync", "team")["action"] == "fetched"
    assert sorted(labels("--store", "team")) == ["Second fact", "Team fact"]
    [row] = run("stores", "list")["stores"]
    assert row["kind"] == "remote" and row["state"] == "ready" and row["fetched_commit"]


def test_committed_project_entry_must_be_cloned_and_pins_identity(tmp_path, isolated, shared):
    _, remote = shared
    run("stores", "add", "team", "--remote", str(remote), "--scope", "project")
    entry = json.loads((isolated / registry.PROJECT_FILE).read_text())
    clone = registry._clone_path(entry["stores"]["team"]["remote"], "propose")
    remotes._remove_tree(clone)
    with pytest.raises(registry.RegistryError, match="stores sync team"):
        labels("--store", "team")
    entry["stores"]["team"]["store_id"] = str(uuid.uuid4())
    (isolated / registry.PROJECT_FILE).write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="identity"):
        run("stores", "sync", "team")
    assert not clone.exists()


def test_replaced_folder_is_refused(tmp_path):
    path = tmp_path / "mine"
    run("stores", "add", "mine", "--path", str(path), "--init")
    remotes._remove_tree(path)
    git_store.init_store(path)
    with pytest.raises(registry.RegistryError, match="identity-mismatch"):
        labels("--store", "mine")


def test_selection_rules_and_validation(tmp_path, isolated):
    with pytest.raises(service.GraphError, match="not both"):
        run("--root", str(tmp_path), "--store", "x", "find")
    with pytest.raises(service.GraphError, match="init requires --root"):
        run("init")
    with pytest.raises(ValueError, match="credentials"):
        run("stores", "add", "bad", "--remote", "https://user:secret@example.com/store.git")
    (isolated / registry.PROJECT_FILE).write_text('{"schema":1,"stores":{},"stores":{}}')
    with pytest.raises(registry.RegistryError, match="duplicate"):
        run("stores", "list")
    (isolated / registry.PROJECT_FILE).write_text('{"schema":1,"stores":{},"default":"missing"}')
    with pytest.raises(registry.RegistryError, match="not registered"):
        run("stores", "list")
