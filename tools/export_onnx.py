"""Export SoftDemod to ONNX and check it against torch with onnxruntime.

    uv run python tools/export_onnx.py models/v3/model.pt models/v3/ft8nn.onnx

ONNX graph: input "features" (B, 79, 48) float32  ->  output "llr" (B, 174) float32.
The features are model.features(): per-symbol 64-pt DFT, bins -4..19, real parts then imag parts,
scaled by sqrt(median bin power). That preprocessing is NOT in the graph; the C side computes it.
LLR sign convention: llr = log P(bit=1)/P(bit=0), so positive means bit 1.

Every LayerNormalization node is then written out as ReduceMean/Sub/Mul/Add/Sqrt/Div (unfuse_layernorm; the same
function, outputs within ~2e-6). ONNX Runtime 1.31.1's LayerNorm kernel is slow at this width (128): it took 36% of
the model's time at batch 32 on one M4 core, more than all the matrix multiplies, and the plain ops run the whole
model 26% faster (22.4 -> 16.5 ms per batch of 32); ONNX Runtime does not fuse them back. --fused-layernorm keeps
the single op.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft8nn import model, nn_metric, synth_bb  # noqa: E402


def sample_features(n, seed):
    """Mixed synthetic inputs: weak AWGN, strong, fading, QRM, gain steps/bursts (v2 training mix)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    y, _, _, _ = synth_bb.make_batch(n, synth_bb.v2_cfg(), "cpu")
    return model.features(y)


def export(net, path, opset):
    dummy = torch.zeros(2, 79, 2 * model.NBINS)
    torch.onnx.export(
        net, (dummy,), path,
        input_names=["features"], output_names=["llr"],
        dynamic_axes={"features": {0: "batch"}, "llr": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )


def unfuse_layernorm(path):
    """Rewrite each LayerNormalization (last axis) as (x - mean) / sqrt(var + eps) * w + b in plain ops, in place."""
    m = onnx.load(path)
    g = m.graph
    nodes = []
    for n in g.node:
        if n.op_type != "LayerNormalization":
            nodes.append(n)
            continue
        attr = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        assert attr.get("axis", -1) == -1, n.name
        x, w, b = n.input
        p = n.name.strip("/").replace("/", "_")
        g.initializer.append(numpy_helper.from_array(np.array(attr.get("epsilon", 1e-5), np.float32), p + "_eps"))
        nodes += [
            helper.make_node("ReduceMean", [x], [p + "_mu"], axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, p + "_mu"], [p + "_d"]),
            helper.make_node("Mul", [p + "_d", p + "_d"], [p + "_d2"]),
            helper.make_node("ReduceMean", [p + "_d2"], [p + "_var"], axes=[-1], keepdims=1),
            helper.make_node("Add", [p + "_var", p + "_eps"], [p + "_ve"]),
            helper.make_node("Sqrt", [p + "_ve"], [p + "_sd"]),
            helper.make_node("Div", [p + "_d", p + "_sd"], [p + "_n"]),
            helper.make_node("Mul", [p + "_n", w], [p + "_nw"]),
            helper.make_node("Add", [p + "_nw", b], [n.output[0]]),
        ]
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def verify(net, path, n=512, seed=7):
    import onnxruntime as ort

    x = sample_features(n, seed)
    with torch.no_grad():
        ref = net(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    diffs = []
    for bs in (1, 37, n):  # several batch sizes, to exercise the dynamic axis
        got = np.concatenate([sess.run(["llr"], {"features": x[i : i + bs].numpy()})[0] for i in range(0, n, bs)])
        diffs.append(float(np.abs(got - ref).max()))
    sign_agree = float((np.sign(got) == np.sign(ref)).mean())
    return diffs, sign_agree, float(np.abs(ref).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--fused-layernorm", action="store_true", help="keep ONNX LayerNormalization (slower in ONNX Runtime)")
    a = ap.parse_args()

    net = nn_metric.load(a.ckpt)
    export(net, a.out, a.opset)
    if not a.fused_layernorm:
        unfuse_layernorm(a.out)
    diffs, agree, mx = verify(net, a.out)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")
    print(f"max |onnx - torch| per batch size (1, 37, 512): {', '.join(f'{d:.2e}' for d in diffs)}"
          f"  (max |llr| {mx:.1f}, sign agreement {agree:.6f})")
    if max(diffs) > a.tol:
        sys.exit(f"FAIL: diff above tolerance {a.tol}")
    print("OK")


if __name__ == "__main__":
    main()
