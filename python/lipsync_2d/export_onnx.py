"""ONNX export (offline + stateful streaming) and FP16 / INT8 quantization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from lipsync_2d.model import LipSyncStudent, StreamingStudentWrapper


def _export_torch_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset_version: int = 17,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use legacy TorchScript exporter (dynamo=False) for clean dynamic-T GRU + onnxruntime quantization.
    torch.onnx.export(
        module,
        args,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    return output_path


def export_offline_onnx(
    model: LipSyncStudent,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export offline batch ONNX: mel [B, T_10ms, 80] -> viseme15, rhubarb9, mouth_curves."""
    model.eval()
    device = next(model.parameters()).device
    dummy_mel = torch.randn(1, 20, model.config.n_mels, device=device, dtype=torch.float32)
    return _export_torch_onnx(
        module=model,
        args=(dummy_mel,),
        output_path=Path(output_path),
        input_names=["mel"],
        output_names=["viseme15_logits", "rhubarb9_logits", "mouth_curves"],
        dynamic_axes={
            "mel": {0: "batch", 1: "t_10ms"},
            "viseme15_logits": {0: "batch", 1: "t_50hz"},
            "rhubarb9_logits": {0: "batch", 1: "t_50hz"},
            "mouth_curves": {0: "batch", 1: "t_50hz"},
        },
        opset_version=opset_version,
    )


def export_streaming_onnx(
    model: LipSyncStudent,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export stateful streaming ONNX: (mel_chunk, h_in, conv_cache_in) -> logits & next state."""
    model.eval()
    device = next(model.parameters()).device
    dummy_mel = torch.randn(1, 4, model.config.n_mels, device=device, dtype=torch.float32)
    h_in, conv_cache_in = model.init_state(batch_size=1, device=device, dtype=torch.float32)
    wrapper = StreamingStudentWrapper(model).eval()
    return _export_torch_onnx(
        module=wrapper,
        args=(dummy_mel, h_in, conv_cache_in),
        output_path=Path(output_path),
        input_names=["mel_chunk", "h_in", "conv_cache_in"],
        output_names=[
            "viseme15_logits",
            "rhubarb9_logits",
            "mouth_curves",
            "h_out",
            "conv_cache_out",
        ],
        dynamic_axes={
            "mel_chunk": {0: "batch", 1: "t_10ms"},
            "h_in": {1: "batch"},
            "conv_cache_in": {0: "batch"},
            "viseme15_logits": {0: "batch", 1: "t_50hz"},
            "rhubarb9_logits": {0: "batch", 1: "t_50hz"},
            "mouth_curves": {0: "batch", 1: "t_50hz"},
            "h_out": {1: "batch"},
            "conv_cache_out": {0: "batch"},
        },
        opset_version=opset_version,
    )


def quantize_onnx_fp16(src_path: str | Path, dst_path: str | Path) -> Path:
    """Convert an FP32 ONNX model to FP16 weights with FP32 I/O casts for broad runtime support."""
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src))
    graph = model.graph

    # Convert float32 initializers (except small normalization/shape/scale tensors) to float16
    # or wrap model graph inputs/outputs with Cast nodes to keep FP32 I/O API identical.
    fp32_to_fp16_names: set[str] = set()
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            fp16_arr = arr.astype(np.float16)
            new_init = numpy_helper.from_array(fp16_arr, name=init.name)
            init.CopyFrom(new_init)
            fp32_to_fp16_names.add(init.name)

    for vi in list(graph.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        elif node.op_type in ("ConstantOfShape",):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))

    # Cast FP32 graph inputs -> FP16 inside graph, and FP16 graph outputs -> FP32
    pre_nodes: list[onnx.NodeProto] = []
    input_remap: dict[str, str] = {}
    for idx, inp in enumerate(graph.input):
        if inp.type.HasField("tensor_type") and inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            fp16_name = f"{inp.name}__fp16_{idx}"
            input_remap[inp.name] = fp16_name
            pre_nodes.append(
                helper.make_node("Cast", inputs=[inp.name], outputs=[fp16_name], to=TensorProto.FLOAT16)
            )

    output_remap: dict[str, str] = {}
    post_nodes: list[onnx.NodeProto] = []
    for idx, out in enumerate(graph.output):
        if out.type.HasField("tensor_type") and out.type.tensor_type.elem_type == TensorProto.FLOAT:
            fp16_name = f"{out.name}__fp16_out_{idx}"
            output_remap[out.name] = fp16_name
            post_nodes.append(
                helper.make_node("Cast", inputs=[fp16_name], outputs=[out.name], to=TensorProto.FLOAT)
            )

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in input_remap:
                node.input[i] = input_remap[name]
        for i, name in enumerate(node.output):
            if name in output_remap:
                node.output[i] = output_remap[name]

    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(pre_nodes + existing_nodes + post_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return dst


def quantize_onnx_int8(src_path: str | Path, dst_path: str | Path) -> Path:
    """Apply dynamic INT8 weight quantization using onnxruntime.quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    return dst


def export_all_variants(
    model: LipSyncStudent,
    output_dir: str | Path,
    prefix: str = "lipsync_student",
) -> dict[str, Path]:
    """Export offline & streaming FP32, FP16, and INT8 ONNX models."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_fp32 = export_offline_onnx(model, out_dir / f"{prefix}_offline_fp32.onnx")
    streaming_fp32 = export_streaming_onnx(model, out_dir / f"{prefix}_streaming_fp32.onnx")

    offline_fp16 = quantize_onnx_fp16(offline_fp32, out_dir / f"{prefix}_offline_fp16.onnx")
    streaming_fp16 = quantize_onnx_fp16(streaming_fp32, out_dir / f"{prefix}_streaming_fp16.onnx")

    offline_int8 = quantize_onnx_int8(offline_fp32, out_dir / f"{prefix}_offline_int8.onnx")
    streaming_int8 = quantize_onnx_int8(streaming_fp32, out_dir / f"{prefix}_streaming_int8.onnx")

    return {
        "offline_fp32": offline_fp32,
        "streaming_fp32": streaming_fp32,
        "offline_fp16": offline_fp16,
        "streaming_fp16": streaming_fp16,
        "offline_int8": offline_int8,
        "streaming_int8": streaming_int8,
    }
