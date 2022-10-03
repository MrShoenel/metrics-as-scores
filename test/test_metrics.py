import base

def test_MetricID():
    from metrics_as_scores.data.metrics import MetricID
    assert len(list(MetricID)) > 0
