REGISTRY = {}

from .basic_controller import BasicMAC
from .mbom_controller import MBOMMAC
from .rbom_controller import RBOMMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['mbom_mac'] = MBOMMAC
REGISTRY['rbom_mac'] = RBOMMAC