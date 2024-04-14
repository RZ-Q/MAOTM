REGISTRY = {}

from .basic_controller import BasicMAC
from .rode_controller import RODEMAC
from .ewm_controller import EWMMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['rode_mac'] = RODEMAC
REGISTRY['EWM_mac'] = EWMMAC