"""
TODO: one-line description of the fdm package
"""

from importlib.metadata import version

__version__ = version("fdm")

# ----- Python unittest-based test framework
# Used for automatic test discovery
from .load_tests import load_tests

# Allow users to run full test suite as fdm.test()
from .test import test
