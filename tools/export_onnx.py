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
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

# ONNX Runtime sends usage telemetry to Microsoft unless this is set before it is loaded (c/src/decode.c)
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft8nn import model, nn_metric, synth_bb  # noqa: E402


def sample_features(n, seed):
    """Mixed synthetic inputs: weak AWGN, strong, fading, QRM, gain steps/bursts (v2 training mix)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    y, _, _, _ = synth_bb.make_batch(n, synth_bb.v2_cfg(), "cpu")
    return model.features(y)


def _contiguous_runs(idxs):
    """Split a sorted index list into [(start, end), ...] half-open contiguous slices."""
    runs = []
    for x in idxs:
        if runs and x == runs[-1][1]:
            runs[-1] = (runs[-1][0], x + 1)
        else:
            runs.append((x, x + 1))
    return runs


def _export_soft_demod_ort(net, path, opset):
    """Build a direct ONNX Runtime graph for SoftDemod:
    - Unfused LayerNorm ops (26% faster than ORT's LayerNormalization at d=128)
    - Tap-wise MatMul + BiasGelu for the 1-D convs (avoids NCL transposes and enables BiasGelu fusion)
    - Fused com.microsoft::Attention (layers 0..L-2) and com.microsoft::MultiHeadAttention on the 58 data_pos
      queries in the final layer (skips dead Costas sync queries and FFN tokens).
    """
    inits, nodes = [], []
    L = int(net.pos.shape[1])
    f_in = int(net.inp.in_features)
    dp = [int(v) for v in net.data_pos.tolist()]
    dp_runs = _contiguous_runs(dp)

    def add_f32(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))
        return name

    def add_i64(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))
        return name

    def add_ln(x, ln_mod, p):
        add_f32(p + "_eps", ln_mod.eps)
        wm = add_f32(p + "_w", ln_mod.weight.detach().cpu().numpy())
        bm = add_f32(p + "_b", ln_mod.bias.detach().cpu().numpy())
        nodes.extend([
            helper.make_node("ReduceMean", [x], [p + "_mu"], axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, p + "_mu"], [p + "_d"]),
            helper.make_node("Mul", [p + "_d", p + "_d"], [p + "_d2"]),
            helper.make_node("ReduceMean", [p + "_d2"], [p + "_var"], axes=[-1], keepdims=1),
            helper.make_node("Add", [p + "_var", p + "_eps"], [p + "_ve"]),
            helper.make_node("Sqrt", [p + "_ve"], [p + "_sd"]),
            helper.make_node("Div", [p + "_d", p + "_sd"], [p + "_n"]),
            helper.make_node("Mul", [p + "_n", wm], [p + "_nw"]),
            helper.make_node("Add", [p + "_nw", bm], [p + "_out"]),
        ])
        return p + "_out"

    def add_conv1d_gelu(x, conv, p):
        w = conv.weight.detach().cpu().numpy()  # (C_out, C_in, K)
        b = conv.bias.detach().cpu().numpy()
        k_sz, dil = conv.kernel_size[0], conv.dilation[0]
        kc = k_sz // 2
        wc = add_f32(f"{p}_t{kc}_w", w[:, :, kc].T)
        acc = f"{p}_t{kc}_mm"
        nodes.append(helper.make_node("MatMul", [x, wc], [acc]))
        ax = add_i64(f"{p}_ax1", [1])
        for k in range(k_sz):
            if k == kc:
                continue
            sh = (k - kc) * dil
            st = f"{p}_t{k}"
            if sh < 0:
                s = -sh
                s0, s1, pads = 0, L - s, [0, s, 0, 0, 0, 0]
            else:
                s = sh
                s0, s1, pads = s, L, [0, 0, 0, 0, s, 0]
            add_i64(st + "_s0", [s0])
            add_i64(st + "_s1", [s1])
            add_i64(st + "_pd", pads)
            wk = add_f32(st + "_w", w[:, :, k].T)
            nodes.append(helper.make_node("Slice", [x, st + "_s0", st + "_s1", ax], [st + "_sl"]))
            nodes.append(helper.make_node("MatMul", [st + "_sl", wk], [st + "_mm"]))
            nodes.append(helper.make_node("Pad", [st + "_mm", st + "_pd"], [st + "_pa"]))
            nodes.append(helper.make_node("Add", [acc, st + "_pa"], [st + "_sum"]))
            acc = st + "_sum"
        nodes.append(helper.make_node("BiasGelu", [acc, add_f32(p + "_b", b)], [p + "_gelu"], domain="com.microsoft"))
        return p + "_gelu"

    def slice_data_pos(src, p):
        ax = add_i64(p + "_ax1", [1])
        outs = []
        for idx, (r0, r1) in enumerate(dp_runs):
            sr = f"{p}_r{idx}"
            add_i64(sr + "_s0", [r0])
            add_i64(sr + "_s1", [r1])
            nodes.append(helper.make_node("Slice", [src, sr + "_s0", sr + "_s1", ax], [sr + "_sl"]))
            outs.append(sr + "_sl")
        if len(outs) == 1:
            return outs[0]
        nodes.append(helper.make_node("Concat", outs, [p + "_cat"], axis=1))
        return p + "_cat"

    wi = add_f32("inp_w", net.inp.weight.detach().cpu().numpy().T)
    bi = add_f32("inp_bp", (net.inp.bias.detach().cpu() + net.pos.detach().cpu()).numpy())
    nodes.append(helper.make_node("MatMul", ["features", wi], ["h0_m"]))
    nodes.append(helper.make_node("Add", ["h0_m", bi], ["h0"]))
    c1 = add_conv1d_gelu("h0", net.conv[0], "c1")
    c2 = add_conv1d_gelu(c1, net.conv[2], "c2")
    nodes.append(helper.make_node("Add", ["h0", c2], ["hc"]))
    h = "hc"

    n_layers = len(net.tf.layers)
    for i, lyr in enumerate(net.tf.layers):
        is_last = (i == n_layers - 1)
        heads = int(lyr.self_attn.num_heads)
        u = add_ln(h, lyr.norm1, f"l{i}_ln1")
        if not is_last:
            wqkv = add_f32(f"l{i}_wqkv", lyr.self_attn.in_proj_weight.detach().cpu().numpy().T)
            bqkv = add_f32(f"l{i}_bqkv", lyr.self_attn.in_proj_bias.detach().cpu().numpy())
            nodes.append(helper.make_node("Attention", [u, wqkv, bqkv], [f"l{i}_att"], domain="com.microsoft", num_heads=heads))
            h_res = h
        else:
            u_q = slice_data_pos(u, "l_last_uq")
            wq, wk, wv = lyr.self_attn.in_proj_weight.detach().cpu().chunk(3, dim=0)
            bq, bk, bv = lyr.self_attn.in_proj_bias.detach().cpu().chunk(3, dim=0)
            nodes.append(helper.make_node("MatMul", [u_q, add_f32(f"l{i}_wq", wq.numpy().T)], [f"l{i}_q"]))
            nodes.append(helper.make_node("MatMul", [u, add_f32(f"l{i}_wk", wk.numpy().T)], [f"l{i}_k"]))
            nodes.append(helper.make_node("MatMul", [u, add_f32(f"l{i}_wv", wv.numpy().T)], [f"l{i}_v"]))
            bqkv = add_f32(f"l{i}_bqkv", np.concatenate([bq.numpy(), bk.numpy(), bv.numpy()]))
            nodes.append(helper.make_node("MultiHeadAttention", [f"l{i}_q", f"l{i}_k", f"l{i}_v", bqkv], [f"l{i}_att"], domain="com.microsoft", num_heads=heads))
            h_res = slice_data_pos(h, "l_last_h")
        wo = add_f32(f"l{i}_wo", lyr.self_attn.out_proj.weight.detach().cpu().numpy().T)
        bo = add_f32(f"l{i}_bo", lyr.self_attn.out_proj.bias.detach().cpu().numpy())
        nodes.append(helper.make_node("MatMul", [f"l{i}_att", wo], [f"l{i}_ao"]))
        nodes.append(helper.make_node("Add", [f"l{i}_ao", bo], [f"l{i}_aob"]))
        nodes.append(helper.make_node("Add", [h_res, f"l{i}_aob"], [f"l{i}_h1"]))
        u2 = add_ln(f"l{i}_h1", lyr.norm2, f"l{i}_ln2")
        w1 = add_f32(f"l{i}_w1", lyr.linear1.weight.detach().cpu().numpy().T)
        b1 = add_f32(f"l{i}_b1", lyr.linear1.bias.detach().cpu().numpy())
        w2 = add_f32(f"l{i}_w2", lyr.linear2.weight.detach().cpu().numpy().T)
        b2 = add_f32(f"l{i}_b2", lyr.linear2.bias.detach().cpu().numpy())
        nodes.append(helper.make_node("MatMul", [u2, w1], [f"l{i}_f1"]))
        nodes.append(helper.make_node("BiasGelu", [f"l{i}_f1", b1], [f"l{i}_fg"], domain="com.microsoft"))
        nodes.append(helper.make_node("MatMul", [f"l{i}_fg", w2], [f"l{i}_f2"]))
        nodes.append(helper.make_node("Add", [f"l{i}_f2", b2], [f"l{i}_f2b"]))
        nodes.append(helper.make_node("Add", [f"l{i}_h1", f"l{i}_f2b"], [f"l{i}_out"]))
        h = f"l{i}_out"

    if n_layers == 0:
        h = slice_data_pos(h, "l0_h")
    uf = add_ln(h, net.norm, "final_ln")
    out_dim = len(dp) * int(net.out.out_features)
    nodes.append(helper.make_node("MatMul", [uf, add_f32("out_w", net.out.weight.detach().cpu().numpy().T)], ["out_m"]))
    nodes.append(helper.make_node("Add", ["out_m", add_f32("out_b", net.out.bias.detach().cpu().numpy())], ["out_mb"]))
    add_i64("out_sh", [-1, out_dim])
    nodes.append(helper.make_node("Reshape", ["out_mb", "out_sh"], ["llr"]))

    g = helper.make_graph(
        nodes, "ft8nn",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, L, f_in])],
        [helper.make_tensor_value_info("llr", TensorProto.FLOAT, [None, out_dim])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 8
    onnx.save(m, path)


def _export_screener_ort(net, path, opset):
    """Build a direct ONNX Runtime graph for model.Screener: input 'features' (B, 79, 48) -> output 'ee' (B,)."""
    inits, nodes = [], []
    L = int(net.pos.shape[1])
    f_in = int(net.inp.in_features)

    def add_f32(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))
        return name

    def add_i64(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))
        return name

    def add_ln(x, ln_mod, p):
        add_f32(p + "_eps", ln_mod.eps)
        wm = add_f32(p + "_w", ln_mod.weight.detach().cpu().numpy())
        bm = add_f32(p + "_b", ln_mod.bias.detach().cpu().numpy())
        nodes.extend([
            helper.make_node("ReduceMean", [x], [p + "_mu"], axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, p + "_mu"], [p + "_d"]),
            helper.make_node("Mul", [p + "_d", p + "_d"], [p + "_d2"]),
            helper.make_node("ReduceMean", [p + "_d2"], [p + "_var"], axes=[-1], keepdims=1),
            helper.make_node("Add", [p + "_var", p + "_eps"], [p + "_ve"]),
            helper.make_node("Sqrt", [p + "_ve"], [p + "_sd"]),
            helper.make_node("Div", [p + "_d", p + "_sd"], [p + "_n"]),
            helper.make_node("Mul", [p + "_n", wm], [p + "_nw"]),
            helper.make_node("Add", [p + "_nw", bm], [p + "_out"]),
        ])
        return p + "_out"

    def add_conv1d_gelu(x, conv, p):
        w = conv.weight.detach().cpu().numpy()
        b = conv.bias.detach().cpu().numpy()
        k_sz, dil = conv.kernel_size[0], conv.dilation[0]
        kc = k_sz // 2
        wc = add_f32(f"{p}_t{kc}_w", w[:, :, kc].T)
        acc = f"{p}_t{kc}_mm"
        nodes.append(helper.make_node("MatMul", [x, wc], [acc]))
        ax = add_i64(f"{p}_ax1", [1])
        for k in range(k_sz):
            if k == kc:
                continue
            sh = (k - kc) * dil
            st = f"{p}_t{k}"
            if sh < 0:
                s = -sh
                s0, s1, pads = 0, L - s, [0, s, 0, 0, 0, 0]
            else:
                s = sh
                s0, s1, pads = s, L, [0, 0, 0, 0, s, 0]
            add_i64(st + "_s0", [s0])
            add_i64(st + "_s1", [s1])
            add_i64(st + "_pd", pads)
            wk = add_f32(st + "_w", w[:, :, k].T)
            nodes.append(helper.make_node("Slice", [x, st + "_s0", st + "_s1", ax], [st + "_sl"]))
            nodes.append(helper.make_node("MatMul", [st + "_sl", wk], [st + "_mm"]))
            nodes.append(helper.make_node("Pad", [st + "_mm", st + "_pd"], [st + "_pa"]))
            nodes.append(helper.make_node("Add", [acc, st + "_pa"], [st + "_sum"]))
            acc = st + "_sum"
        nodes.append(helper.make_node("BiasGelu", [acc, add_f32(p + "_b", b)], [p + "_gelu"], domain="com.microsoft"))
        return p + "_gelu"

    wi = add_f32("inp_w", net.inp.weight.detach().cpu().numpy().T)
    bi = add_f32("inp_bp", (net.inp.bias.detach().cpu() + net.pos.detach().cpu()).numpy())
    nodes.append(helper.make_node("MatMul", ["features", wi], ["h0_m"]))
    nodes.append(helper.make_node("Add", ["h0_m", bi], ["h0"]))
    c1 = add_conv1d_gelu("h0", net.conv[0], "c1")
    c2 = add_conv1d_gelu(c1, net.conv[2], "c2")
    nodes.append(helper.make_node("Add", ["h0", c2], ["hc"]))
    h = "hc"

    for i, lyr in enumerate(net.tf.layers):
        heads = int(lyr.self_attn.num_heads)
        u = add_ln(h, lyr.norm1, f"l{i}_ln1")
        wqkv = add_f32(f"l{i}_wqkv", lyr.self_attn.in_proj_weight.detach().cpu().numpy().T)
        bqkv = add_f32(f"l{i}_bqkv", lyr.self_attn.in_proj_bias.detach().cpu().numpy())
        nodes.append(helper.make_node("Attention", [u, wqkv, bqkv], [f"l{i}_att"], domain="com.microsoft", num_heads=heads))
        wo = add_f32(f"l{i}_wo", lyr.self_attn.out_proj.weight.detach().cpu().numpy().T)
        bo = add_f32(f"l{i}_bo", lyr.self_attn.out_proj.bias.detach().cpu().numpy())
        nodes.append(helper.make_node("MatMul", [f"l{i}_att", wo], [f"l{i}_ao"]))
        nodes.append(helper.make_node("Add", [f"l{i}_ao", bo], [f"l{i}_aob"]))
        nodes.append(helper.make_node("Add", [h, f"l{i}_aob"], [f"l{i}_h1"]))
        u2 = add_ln(f"l{i}_h1", lyr.norm2, f"l{i}_ln2")
        w1 = add_f32(f"l{i}_w1", lyr.linear1.weight.detach().cpu().numpy().T)
        b1 = add_f32(f"l{i}_b1", lyr.linear1.bias.detach().cpu().numpy())
        w2 = add_f32(f"l{i}_w2", lyr.linear2.weight.detach().cpu().numpy().T)
        b2 = add_f32(f"l{i}_b2", lyr.linear2.bias.detach().cpu().numpy())
        nodes.append(helper.make_node("MatMul", [u2, w1], [f"l{i}_f1"]))
        nodes.append(helper.make_node("BiasGelu", [f"l{i}_f1", b1], [f"l{i}_fg"], domain="com.microsoft"))
        nodes.append(helper.make_node("MatMul", [f"l{i}_fg", w2], [f"l{i}_f2"]))
        nodes.append(helper.make_node("Add", [f"l{i}_f2", b2], [f"l{i}_f2b"]))
        nodes.append(helper.make_node("Add", [f"l{i}_h1", f"l{i}_f2b"], [f"l{i}_out"]))
        h = f"l{i}_out"

    uf = add_ln(h, net.norm, "final_ln")
    nodes.append(helper.make_node("ReduceMean", [uf], ["p_mean"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", [uf], ["p_max"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("Concat", ["p_mean", "p_max"], ["pooled"], axis=-1))
    hw0 = add_f32("hd0_w", net.head[0].weight.detach().cpu().numpy().T)
    hb0 = add_f32("hd0_b", net.head[0].bias.detach().cpu().numpy())
    hw2 = add_f32("hd2_w", net.head[2].weight.detach().cpu().numpy().T)
    hb2 = add_f32("hd2_b", net.head[2].bias.detach().cpu().numpy())
    nodes.append(helper.make_node("MatMul", ["pooled", hw0], ["hd0_m"]))
    nodes.append(helper.make_node("BiasGelu", ["hd0_m", hb0], ["hd0_g"], domain="com.microsoft"))
    nodes.append(helper.make_node("MatMul", ["hd0_g", hw2], ["hd2_m"]))
    nodes.append(helper.make_node("Add", ["hd2_m", hb2], ["hd2_mb"]))
    add_i64("ee_sh", [-1])
    nodes.append(helper.make_node("Reshape", ["hd2_mb", "ee_sh"], ["ee"]))

    g = helper.make_graph(
        nodes, "ft8nn_screener",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, L, f_in])],
        [helper.make_tensor_value_info("ee", TensorProto.FLOAT, [None])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 8
    onnx.save(m, path)


def export(net, path, opset, fused_layernorm=False):
    if isinstance(net, model.Screener) and not fused_layernorm:
        _export_screener_ort(net, path, opset)
        return
    if isinstance(net, model.SoftDemod) and not fused_layernorm:
        _export_soft_demod_ort(net, path, opset)
        return
    dummy = torch.zeros(2, 79, 2 * model.NBINS)
    torch.onnx.export(
        net, (dummy,), path,
        input_names=["features"], output_names=["llr"],
        dynamic_axes={"features": {0: "batch"}, "llr": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    if not fused_layernorm:
        unfuse_layernorm(path)


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
    out_name = sess.get_outputs()[0].name
    diffs = []
    for bs in (1, 37, n):  # several batch sizes, to exercise the dynamic axis
        got = np.concatenate([sess.run([out_name], {"features": x[i : i + bs].numpy()})[0] for i in range(0, n, bs)])
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
    export(net, a.out, a.opset, fused_layernorm=a.fused_layernorm)
    diffs, agree, mx = verify(net, a.out)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")
    print(f"max |onnx - torch| per batch size (1, 37, 512): {', '.join(f'{d:.2e}' for d in diffs)}"
          f"  (max |llr| {mx:.1f}, sign agreement {agree:.6f})")
    if max(diffs) > a.tol:
        sys.exit(f"FAIL: diff above tolerance {a.tol}")

    ckpt_p, out_p = Path(a.ckpt), Path(a.out)
    if ckpt_p.name != "screener.pt" and out_p.name != "screener.onnx" and not a.fused_layernorm:
        scr_pt = ckpt_p.parent / "screener.pt"
        scr_onnx = out_p.parent / "screener.onnx"
        if scr_pt.exists():
            snet = nn_metric.load(scr_pt)
            export(snet, scr_onnx, a.opset)
            sdiffs, _, _ = verify(snet, scr_onnx)
            if max(sdiffs) > a.tol:
                sys.exit(f"FAIL: screener diff {max(sdiffs):.2e} above tolerance {a.tol}")
            print(f"wrote {scr_onnx} ({scr_onnx.stat().st_size / 1e6:.2f} MB, max diff {max(sdiffs):.2e})")
        elif scr_onnx.exists() and ckpt_p.parent.resolve() != out_p.parent.resolve():
            scr_onnx.unlink()
    print("OK")


if __name__ == "__main__":
    main()
