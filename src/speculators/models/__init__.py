from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .dfly import DFlyDraftModel, DFlySpeculatorConfig
from .domino import DominoDraftModel, DominoSpeculatorConfig
from .dspark import DSparkDraftModel, DSparkSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .kv_native_dflash import (
    KVNativeDFlashDraftModel,
    KVNativeDFlashSpeculatorConfig,
)
from .mtp import MTPDraftModel, MTPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig

__all__ = [
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "DFlyDraftModel",
    "DFlySpeculatorConfig",
    "DSparkDraftModel",
    "DSparkSpeculatorConfig",
    "DominoDraftModel",
    "DominoSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "KVNativeDFlashDraftModel",
    "KVNativeDFlashSpeculatorConfig",
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
]
