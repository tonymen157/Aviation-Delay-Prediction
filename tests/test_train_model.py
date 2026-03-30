"""Unit tests for src/models/train_model.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# ----------------------------------------------------------------------
# Adjust import path so we can import the script as a module
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src/models')
)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Mock heavy dependencies that would cause import errors
EXTERNAL_PACKAGES = {
    "joblib": MagicMock(),
    "lightgbm": MagicMock(),
    "pandas": MagicMock(),
    "sklearn": MagicMock(),
}
for name in EXTERNAL_PACKAGES:
    sys.modules[name] = EXTERNAL_PACKAGES[name]

# Mock any utility functions (e.g., logging) if needed
with patch('src.models.train_model.setup_logger'), \
     patch('src.models.train_model.Path'), \
     patch('src.models.train_model.pl'):

    # Import the target module under test
    import train_model as tm


class TestTrainModelSanity(unittest.TestCase):
    """Basic sanity‑checks – does the file import and expose the expected public API?"""

    def test_import_succeeds(self):
        """Importing the module should not raise an exception."""
        self.assertTrue(hasattr(tm, "main"))
        self.assertTrue(callable(tm.main))

    def test_main_exists(self):
        """main() must be defined."""
        self.assertTrue(hasattr(tm, "main"))
        self.assertTrue(callable(tm.main))


if __name__ == '__main__':
    unittest.main()