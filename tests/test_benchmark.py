from benchmarks.run import run


def test_benchmark_uses_disposable_valid_graph():
    result = run(4, 1, 1)
    assert result["nodes"] == 4
    assert len(result["corpus_sha256"]) == 64
    assert len(result["seconds"]["samples"]) == 1
    assert result["response_bytes"][0] > 0
    assert result["status"] == "measured_not_release_gated"


def test_benchmark_separates_cold_and_warm_samples():
    result = run(4, 3, 0, cold_iterations=2)
    assert len(result["seconds"]["samples"]) == 3
    assert result["seconds"]["cold_p95"] is not None
    assert result["seconds"]["warm_p95"] is not None


def test_benchmark_can_measure_bounded_traversal():
    result = run(4, 2, 0, workload="neighbors")
    assert result["workload"] == "neighbors"
    assert len(result["seconds"]["samples"]) == 2


def test_benchmark_can_measure_exact_tag_filtering():
    assert run(4, 1, 0, workload="tag")["workload"] == "tag"
