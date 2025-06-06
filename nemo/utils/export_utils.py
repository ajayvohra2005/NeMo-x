# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import nullcontext
from enum import Enum
from typing import Callable, Dict, Optional, Type

from megatron.core.device_utils import get_current_device_type
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import CastToFloat, CastToFloatAll, logging

try:
    import onnxruntime

    ort_available = True
except (ImportError, ModuleNotFoundError):
    ort_available = False


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = 1
    TORCHSCRIPT = 2


_EXT_DICT = {
    ".pt": ExportFormat.TORCHSCRIPT,
    ".ts": ExportFormat.TORCHSCRIPT,
    ".onnx": ExportFormat.ONNX,
}


class TorchRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        """
        LayerNorm without bias
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # can be only calculated with precision=32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LinearWithBiasSkip(nn.Module):
    def __init__(self, weight, bias, skip_bias_add):
        super(LinearWithBiasSkip, self).__init__()
        self.bias = bias
        self.weight = weight
        self.skip_bias_add = skip_bias_add

    def forward(self, x, weight=None):
        if weight is None:
            weight = self.weight
        if self.skip_bias_add:
            return F.linear(x, weight), self.bias
        return F.linear(x, weight, self.bias), None


def get_export_format(filename: str):
    _, ext = os.path.splitext(filename)
    try:
        return _EXT_DICT[ext.lower()]
    except KeyError:
        raise ValueError(f"Export file {filename} extension does not correspond to any export format!")


def augment_filename(output: str, prepend: str):
    if prepend == 'self':
        return output

    path, filename = os.path.split(output)
    filename = f"{prepend}-{filename}"
    return os.path.join(path, filename)


def forward_method(self):
    if hasattr(self, "forward_for_export"):
        return self.forward_for_export
    else:
        return self.forward


def wrap_forward_method(self):
    tp = type(self)
    old_forward_method = None
    if hasattr(tp, "forward_for_export"):
        forward_method = tp.forward_for_export
        old_forward_method = tp.forward
        tp.forward = forward_method
    else:
        forward_method = None
    return forward_method, old_forward_method


def parse_input_example(input_example):
    input_list = list(input_example)
    input_dict = {}
    # process possible kwargs
    if isinstance(input_list[-1], dict):
        input_dict = input_list[-1]
        input_list = input_list[:-1]
    return input_list, input_dict


def to_onnxrt_input(ort_input_names, input_names, input_dict, input_list):
    odict = {}
    if not input_names:
        input_list.extend(input_dict.values())
        for k, v in zip(ort_input_names, input_list):
            odict[k] = v.cpu().numpy()
        return odict
    for k in reversed(input_names):
        val = None
        if k in input_dict:
            val = input_dict[k].cpu().numpy()
        elif len(input_list) > 0:
            val = input_list.pop().cpu().numpy()
        if k in ort_input_names and val is not None:
            odict[k] = val
    return odict


def verify_torchscript(model, output, input_examples, check_tolerance=0.01):
    all_good = True
    for input_example in input_examples:
        input_list, input_dict = parse_input_example(input_example)
        # We disable autocast here to make sure exported TS will run under Triton or other C++ env
        with torch.amp.autocast(get_current_device_type(), enabled=False):
            output_example = model.forward(*input_list, **input_dict)
            ts_model = torch.jit.load(output)
            all_good = all_good and run_ts_and_compare(
                ts_model, input_list, input_dict, output_example, check_tolerance
            )
    status = "SUCCESS" if all_good else "FAIL"
    logging.info(f"Torchscript generated at {output} verified with torchscript forward : " + status)
    return all_good


def verify_runtime(model, output, input_examples, input_names, check_tolerance=0.01):
    onnx_model = onnx.load(output)
    ort_input_names = [node.name for node in onnx_model.graph.input]

    global ort_available
    if not ort_available:
        logging.warning(f"ONNX generated at {output}, not verified - please install onnxruntime_gpu package.\n")
        onnx.checker.check_model(onnx_model, full_check=True)
        return
    onnx_session_opt = onnxruntime.SessionOptions()
    onnx_session_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), sess_options=onnx_session_opt, providers=['CUDAExecutionProvider']
    )
    del onnx_model
    all_good = True
    for input_example in input_examples:
        input_list, input_dict = parse_input_example(input_example)
        output_example = model.forward(*input_list, **input_dict)
        if not isinstance(output_example, tuple):
            output_example = (output_example,)
        ort_input = to_onnxrt_input(ort_input_names, input_names, input_dict, input_list)
        all_good = all_good and run_ort_and_compare(sess, ort_input, output_example, check_tolerance)
    status = "SUCCESS" if all_good else "FAIL"
    logging.info(f"ONNX generated at {output} verified with onnxruntime : " + status)
    return all_good


def run_ts_and_compare(ts_model, ts_input_list, ts_input_dict, output_example, check_tolerance=0.01):
    # Verify the model can be read, and is valid
    ts_out = ts_model(*ts_input_list, **ts_input_dict)

    all_good = True
    for i, out in enumerate(ts_out):
        expected = output_example[i]

        if torch.is_tensor(expected):
            tout = out.to('cpu')
            logging.debug(f"Checking output {i}, shape: {expected.shape}:\n")
            this_good = True
            try:
                if not torch.allclose(tout, expected.cpu(), rtol=check_tolerance, atol=check_tolerance):
                    this_good = False
            except Exception:  # there may ne size mismatch and it may be OK
                this_good = False
            if not this_good:
                logging.info(f"Results mismatch! PyTorch(expected):\n{expected}\nTorchScript:\n{tout}")
                all_good = False
    return all_good


def run_ort_and_compare(sess, ort_input, output_example, check_tolerance=0.01):
    # Verify the model can be read, and is valid
    ort_out = sess.run(None, ort_input)
    all_good = True
    for i, out in enumerate(ort_out):
        expected = output_example[i]

        if torch.is_tensor(expected):
            tout = torch.from_numpy(out)
            logging.debug(f"Checking output {i}, shape: {expected.shape}:\n")
            this_good = True
            try:
                if not torch.allclose(tout, expected.cpu(), rtol=check_tolerance, atol=100 * check_tolerance):
                    this_good = False
            except Exception:  # there may be size mismatch and it may be OK
                this_good = False
            if not this_good:
                logging.info(
                    f"onnxruntime results mismatch! PyTorch(expected, {expected.shape}):\n{expected}\nONNXruntime, {tout.shape}:\n{tout}"
                )
                all_good = False
    return all_good


apex_available = True

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    from apex.normalization import MixedFusedRMSNorm
    from apex.normalization.fused_layer_norm import FusedLayerNorm, MixedFusedLayerNorm
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm as MCoreFusedLayerNorm
    from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    def replace_FusedLayerNorm(n: nn.Module) -> Optional[nn.LayerNorm]:
        """
        Replaces Apex's FusedLayerNorm with nn.LayerNorm. This is required for ONNX export.
        Args:
           n: the FusedLayerNorm pytorch module to replace
        Returns:
           Equivalent LayerNorm module
        """

        p = next(n.parameters())

        if isinstance(n, FusedLayerNorm) or isinstance(n, MixedFusedLayerNorm):
            shape, eps, affine = n.normalized_shape, n.eps, n.elementwise_affine
        elif isinstance(n, MCoreFusedLayerNorm):
            shape, eps, affine = n.weight.shape, n.eps, True
        elif isinstance(n, FastLayerNorm):
            shape, eps, affine = n.weight.shape, n.epsilon, True
        else:
            return None

        n_state = n.state_dict()
        mod = nn.LayerNorm(shape, eps=eps, elementwise_affine=affine, device=p.device, dtype=p.dtype)

        mod.load_state_dict(n_state, strict=True)

        return mod

    def replace_MixedFusedRMSNorm(n: nn.Module):
        """
        Replaces Apex's MixedFusedRMSNorm with equivalent Pytorch layer. This is required for ONNX export.
        Args:
           n: the MixedFusedRMSNorm pytorch module to replace
        Returns:
           Equivalent module
        """

        p = next(n.parameters())

        if isinstance(n, MixedFusedRMSNorm):
            mod = TorchRMSNorm(n.state_dict()['weight'], n.eps).to(p.device)
        else:
            return None

        return mod

    def replace_ParallelLinear(n: nn.Module) -> Optional[nn.Linear]:
        """
        Replaces Apex's ColumnParallelLinear or RowParallelLinear with nn.Linear
        Args:
           n: the nn.Module pytorch module to replace
        Returns:
           Equivalent Linear module
        """
        if not (isinstance(n, ColumnParallelLinear) or isinstance(n, RowParallelLinear)):
            raise ValueError("This function can only change the ColumnParallelLinear or RowParallelLinear module.")

        dev = next(n.parameters()).device
        mod = LinearWithBiasSkip(n.weight, n.bias, n.skip_bias_add).to(dev)

        n_state = n.state_dict()
        mod.load_state_dict(n_state, strict=False)
        return mod

    def replace_FusedScaleMaskSoftmax(n: nn.Module) -> Optional[nn.Linear]:
        """
        Replaces Apex's FusedScaleMaskSoftmax with nn.LayerNorm. This is required for ONNX export.
        Args:
           n: the FusedScaleMaskSoftmax module to replace
        Returns:
           Equivalent LayerNorm module
        """
        if not isinstance(n, FusedScaleMaskSoftmax):
            logging.warning(f"This function can only change the FusedScaleMaskSoftmax module, got: {n.__class__}")
            return n

        # disable the fusion only
        mod = FusedScaleMaskSoftmax(
            n.input_in_fp16, n.input_in_bf16, n.attn_mask_type, False, n.mask_func, n.softmax_in_fp32, n.scale
        )

        return mod

    default_Apex_replacements = {
        "FusedLayerNorm": replace_FusedLayerNorm,
        "MixedFusedLayerNorm": replace_FusedLayerNorm,
        "MCoreFusedLayerNorm": replace_FusedLayerNorm,
        "FastLayerNorm": replace_FusedLayerNorm,
        "RowParallelLinear": replace_ParallelLinear,
        "ColumnParallelLinear": replace_ParallelLinear,
        "FusedScaleMaskSoftmax": replace_FusedScaleMaskSoftmax,
        "MixedFusedRMSNorm": replace_MixedFusedRMSNorm,
    }

except Exception as e:
    default_Apex_replacements = {}
    apex_available = False


def simple_replace(BaseT: Type[nn.Module], DestT: Type[nn.Module]) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT. BaseT and DestT should have same atrributes. No weights are copied.
    Args:
        BaseT : module type to replace
        DestT : destination module type
    Returns:
        swap function to replace BaseT module with DestT
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        if not isinstance(mod, BaseT):
            return None
        args = [getattr(mod, name, None) for name in mod.__constants__]
        out = DestT(*args)
        return out

    return expansion_fn


def replace_MatchedScaleMaskSoftmax(n: nn.Module) -> Optional[nn.Linear]:
    """
    Replaces MatchedScaleMaskSoftmax with exportable softmax layer
    Args:
        n: module to replace
    Returns:
        exportable module
    """
    # including the import here to avoid circular imports
    from nemo.collections.nlp.modules.common.megatron.fused_softmax import MatchedScaleMaskSoftmax

    # disabling fusion for the MatchedScaleMaskSoftmax
    mod = MatchedScaleMaskSoftmax(
        n.input_in_fp16, n.input_in_bf16, n.attn_mask_type, False, n.mask_func, n.softmax_in_fp32, n.scale
    )
    return mod


def wrap_module(BaseT: Type[nn.Module], DestT: Type[nn.Module]) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT wrapper.
    Args:
        BaseT : module type to replace
        DestT : destination module type
    Returns:
        swap function to replace BaseT module with DestT
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        out = DestT(mod)
        return out

    return expansion_fn


def swap_modules(model: nn.Module, mapping: Dict[str, nn.Module]):
    """
    This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children

    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            parent_mod = parent_mod._modules[sub_path]  # noqa
        parent_mod._modules[expanded_path[-1]] = new_mod  # noqa

    return model


def replace_modules(
    model: nn.Module, expansions: Dict[str, Callable[[nn.Module], Optional[nn.Module]]] = None
) -> nn.Module:
    """
    Top-level function to replace modules in model, specified by class name with a desired replacement.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        expansions : replacement dictionary: module class name -> replacement function generator
    Returns:
        model, possibly modified in-place
    """
    mapping: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        m_type = type(m).__name__
        if m_type in expansions:
            swapped = expansions[m_type](m)
            if swapped:
                mapping[name] = swapped
    if len(mapping) > 0:
        logging.info(f"Swapped {len(mapping)} modules")
    swap_modules(model, mapping)
    return model


def script_module(m: nn.Module):
    return torch.jit.script(m)


script_replacements = {}


def replace_for_export(model: nn.Module) -> nn.Module:
    """
    Top-level function to replace 'default set' of modules in model, called from _prepare_for_export.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
    Returns:
        model, possibly modified in-place
    """
    default_replacements = {
        "MatchedScaleMaskSoftmax": wrap_module(None, replace_MatchedScaleMaskSoftmax),
    }

    replace_modules(model, default_Apex_replacements)
    replace_modules(model, default_replacements)
    # This one has to be the last
    replace_modules(model, script_replacements)


def add_casts_around_norms(model: nn.Module):
    """
    Function to put additional to/from float32 casts around operations known to require full precision.
    It was used with an extra post-parse script to have TRT preserve extra precision when --fp16 needed.
    Should not be needed with TRT 8.6.1 or later.
    """
    from nemo.collections.tts.modules.submodules import MaskedInstanceNorm1d

    default_cast_replacements = {
        "BatchNorm1d": wrap_module(nn.BatchNorm1d, CastToFloat),
        "BatchNorm2d": wrap_module(nn.BatchNorm2d, CastToFloat),
        "LayerNorm": wrap_module(nn.LayerNorm, CastToFloat),
        "InstanceNorm1d": wrap_module(nn.InstanceNorm1d, CastToFloat),
        "MaskedInstanceNorm1d": wrap_module(MaskedInstanceNorm1d, CastToFloatAll),
    }
    replace_modules(model, default_cast_replacements)


def rename_onnx_io(output, input_names, output_names):
    onnx_model = onnx.load(output)
    rename_map = {}
    for inp, name in zip(onnx_model.graph.input, input_names):
        rename_map[inp.name] = name
    for out, name in zip(onnx_model.graph.output, output_names):
        rename_map[out.name] = name
    for n in onnx_model.graph.node:
        for inp in range(len(n.input)):
            if n.input[inp] in rename_map:
                n.input[inp] = rename_map[n.input[inp]]
        for out in range(len(n.output)):
            if n.output[out] in rename_map:
                n.output[out] = rename_map[n.output[out]]

    for i in range(len(input_names)):
        onnx_model.graph.input[i].name = input_names[i]
    for i in range(len(output_names)):
        onnx_model.graph.output[i].name = output_names[i]
    onnx.save(onnx_model, output)
