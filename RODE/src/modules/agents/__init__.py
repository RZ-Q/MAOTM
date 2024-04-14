REGISTRY = {}

from .rnn_agent import RNNAgent
from .rode_agent import RODEAgent
from .ewm_agent import EWMAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rode"] = RODEAgent
REGISTRY["ewm"] = EWMAgent
