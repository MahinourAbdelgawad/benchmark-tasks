"""Use the real yam_arena when it is installed (the cloud machine); otherwise the local stand-in in _shim/."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
if importlib.util.find_spec("yam_arena") is None:
    sys.path.insert(0, str(HERE / "_shim"))
