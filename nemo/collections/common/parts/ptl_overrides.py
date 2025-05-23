# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin
from megatron.core.device_utils import get_current_device_type


class NeMoMixedPrecisionPlugin(MixedPrecisionPlugin):
    def __init__(self, init_scale: float = 2**32, growth_interval: int = 1000) -> None:
        super().__init__(precision=16)

        self.scaler = torch.amp.GradScaler(get_current_device_type(), init_scale=init_scale, growth_interval=growth_interval)
