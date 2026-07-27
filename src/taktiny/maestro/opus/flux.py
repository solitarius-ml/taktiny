


from taktiny.cosettes.schedulers.euler.flow_match_discrete import FlowMatchEulerDiscreteScheduler
from taktiny.cosettes.transformers.flux import Flux2Transformer2DModel
from taktiny.cosettes.autoencoders.flux import AutoencoderKLFlux2
from taktiny.cosettes._common import DiffusionIM


class Flux2(DiffusionIM):
    def __init__(self):
        super().__init__()
