# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Flux model"""

from taktiny.cosettes.schedulers.euler.flow_match_discrete import FlowMatchEulerDiscreteScheduler
from taktiny.cosettes.transformers.flux import Flux2Transformer2DModel
from taktiny.cosettes.autoencoders.flux import AutoencoderKLFlux2
from taktiny.cosettes._common import DiffusionIM


class Flux2(DiffusionIM):
    def __init__(self):
        super().__init__()
