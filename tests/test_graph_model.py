import tempfile
import uuid
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from memory_graph import git_store, service


def uid():
    return str(uuid.uuid4())


@settings(max_examples=12, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3), st.booleans()), max_size=10))
def test_link_sequences_match_independent_graph_model(operations):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "store"
        git_store.init_store(root)
        ids = [uid() for _ in range(4)]
        service.mutate(
            root,
            uid(),
            "apply",
            {
                "changes": [
                    {
                        "op": "create",
                        "node": {"id": node, "label": f"Node {index}", "kind": "arbitrary"},
                    }
                    for index, node in enumerate(ids)
                ]
            },
        )
        model = {node: set() for node in ids}
        for source_index, target_index, add in operations:
            source, target = ids[source_index], ids[target_index]
            state = service.show(root, source)
            edge = {"label": "unfamiliar-relationship", "to": target}
            request = {"id": source, "expected_revision": state["revision"], "edge": edge}
            if add:
                service.mutate(root, uid(), "link", request)
                model[source].add(target)
            elif target in model[source]:
                service.mutate(root, uid(), "unlink", request)
                model[source].remove(target)
            actual = service.show(root, source)["node"]["edges"]
            assert {edge["to"] for edge in actual} == model[source]
        for source in ids:
            result = service.neighbors(root, source, "out", None, None, False, 1, 50)
            assert {item["id"] for item in result["results"]} == model[source] - {source}
            assert {
                (edge["source"], edge["label"], edge["target"]) for edge in result["relationships"]
            } == {(source, "unfamiliar-relationship", target) for target in model[source]}
