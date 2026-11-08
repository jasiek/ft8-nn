from pathlib import Path

import pytest

CKPT = Path(__file__).resolve().parents[1] / "models/v2/model.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="models/v2/model.pt not present")
def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    import sys

    sys.path.insert(0, str(CKPT.parents[2] / "tools"))
    import export_onnx
    from ft8nn import nn_metric

    net = nn_metric.load(CKPT)
    out = tmp_path / "m.onnx"
    export_onnx.export(net, out, 17)
    diffs, agree, _ = export_onnx.verify(net, out, n=64)
    assert max(diffs) < 1e-3
    assert agree == 1.0
