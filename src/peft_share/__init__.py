"""peft-share: Compress multiple PEFT LoRA adapters into a shared subspace."""

from peft_share.config import SHAREConfig
from peft_share.model import SHAREModel

__version__ = "0.1.0"
__all__ = ["SHAREModel", "SHAREConfig"]
