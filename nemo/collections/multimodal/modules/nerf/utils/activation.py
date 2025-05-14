# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from torch.autograd import Function
from functools import partial
from megatron.core.device_utils import get_current_device_type
custom_fwd = partial(torch.amp.custom_fwd, device_type=get_current_device_type())
custom_bwd = partial(torch.amp.custom_bwd, device_type=get_current_device_type())


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15))


trunc_exp = _trunc_exp.apply
