"""
Common constants for the MLprints library.
"""

import os
from pathlib import Path


# cache directory (to be set by the user via the MLPRINTS_HOME environment variable)
mlprints_home = os.getenv("MLPRINTS_HOME")
# standard linux cache directory (fallback if MLPRINTS_HOME is not set)
xdg_cache_home = os.getenv("XDG_CACHE_HOME")

MLPRINTS_HOME = None
if mlprints_home is not None:
    MLPRINTS_HOME = Path(os.path.expandvars(mlprints_home)).expanduser()
elif xdg_cache_home is not None:
    MLPRINTS_HOME = Path(os.path.expandvars(xdg_cache_home)).expanduser() / "mlprints"
else:
    # last fallback for cache directory: the user's home directory
    MLPRINTS_HOME = Path.home() / ".cache" / "mlprints"

COMMON_CACHE_DIR = MLPRINTS_HOME / "common"
FINGERPRINT_CACHE_DIR = MLPRINTS_HOME / "fingerprint"
ATTACK_CACHE_DIR = MLPRINTS_HOME / "attack"

# inference and training
MASK_LOSS_ID = -100
MASK_NO_MODIFIABLE_ID = -100
MAX_LENGTH_SENTINEL = int(10**7)
