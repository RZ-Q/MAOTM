REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .policy_inference_agent import PIAgent
REGISTRY["policy_infer"] = PIAgent

# from .mbom_agent_new import MBOMAgent
# REGISTRY['mbom'] = MBOMAgent

from .mbom_agent import MBOMAgent
REGISTRY['mbom'] = MBOMAgent

from .rbom_agent import RBOMAgent
REGISTRY['rbom'] = RBOMAgent