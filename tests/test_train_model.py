"""Unit tests for src/models/train_model.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

class TestTrainModelSanity(unittest.TestCase):
    """Basic sanity‑checks – does the file import and expose the expected public API?"""

    @patch.dict('sys.modules', {
        'joblib': MagicMock(),
        'lightgbm': MagicMock(),
        'pandas': MagicMock(),
        'sklearn': MagicMock(),
        'sklearn.model_selection': MagicMock(),
        'sklearn.metrics': MagicMock(),
        'utils': MagicMock(),
        'utils.logging_config': MagicMock(),
    })
    def test_import_succeeds(self):
        """Importing the module should not raise an exception."""
        # Import the module inside the test method so that the mocked modules are in place
        from src.models import train_model
        # Mock utility functions used in the module (if needed for import)
        with patch.object(train_model, 'setup_logger'), \
             patch.object(train_model, 'Path'), \
             patch.object(train_model, 'pl'):
            self.assertTrue(hasattr(train_model, "main"))
            self.assertTrue(callable(train_model.main))

    @patch.dict('sys.modules', {
        'joblib': MagicMock(),
        'lightgbm': MagicMock(),
        'pandas': MagicMock(),
        'sklearn': MagicMock(),
        'sklearn.model_selection': MagicMock(),
        'sklearn.metrics': MagicMock(),
        'utils': MagicMock(),
        'utils.logging_config': MagicMock(),
    })
    def test_main_exists(self):
        """main() must be defined."""
        from src.models import train_model
        with patch.object(train_model, 'setup_logger'), \
             patch.object(train_model, 'Path'), \
             patch.object(train_model, 'pl'):
            self.assertTrue(hasattr(train_model, "main"))
            self.assertTrue(callable(train_model.main))


if __name__ == '__main__':
    unittest.main()