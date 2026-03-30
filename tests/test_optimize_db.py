"""Unit tests for src/scripts/optimize_db.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# ----------------------------------------------------------------------
# Adjust import path so we can import the script as a module
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src/scripts')
)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Mock heavy dependencies that would cause import errors
EXTERNAL_PACKAGES = {
    "pandas": MagicMock(),
    "sqlalchemy": MagicMock(),
    "sqlalchemy.engine": MagicMock(),
}
for name in EXTERNAL_PACKAGES:
    sys.modules[name] = EXTERNAL_PACKAGES[name]

# Mock environment loading utilities
with patch('src.scripts.optimize_db.load_environment_variables'), \
     patch('src.scripts.optimize_db.create_database_engine') as mock_create_engine:
    import optimize_db as od


class TestOptimizeDBImport(unittest.TestCase):
    """Basic sanity checks – the module must import and expose expected symbols."""

    def test_import_succeeds(self):
        """Importing the module must not raise an exception."""
        self.assertTrue(hasattr(od, "main"))
        self.assertTrue(callable(od.main))

    def test_load_dimension_tables_exists(self):
        """load_dimension_tables function must be defined."""
        self.assertTrue(hasattr(od, "load_dimension_tables"))
        self.assertTrue(callable(od.load_dimension_tables))

    def test_optimize_database_exists(self):
        """optimize_database function must be defined."""
        self.assertTrue(hasattr(od, "optimize_database"))
        self.assertTrue(callable(od.optimize_database))

    def test_verify_optimizations_exists(self):
        """verify_optimizations function must be defined."""
        self.assertTrue(hasattr(od, "verify_optimizations"))
        self.assertTrue(callable(od.verify_optimizations))


class TestMainFunction(unittest.TestCase):
    """Verify that the CLI entry‑point correctly calls the library functions."""

    @patch('src.scripts.optimize_db.verify_optimizations')
    @patch('src.scripts.optimize_db.optimize_database')
    @patch('src.scripts.optimize_db.load_dimension_tables')
    @patch('src.scripts.optimize_db.create_database_engine')
    def test_main_calls_expected_functions(self, mock_engine, mock_load_dim, mock_optimize, mock_verify):
        """
        When `main()` runs it should, in order:
        1. Load environment variables.
        2. Create a DB engine.
        3. Load dimension tables.
        4. Run optimization SQL commands.
        5. Verify the optimizations.
        """
        # Make the mock objects return sensible defaults
        mock_conn = MagicMock()
        mock_engine.return_value = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.commit = MagicMock()
        mock_conn.execute = MagicMock()

        # Execute the main routine
        od.main()

        # Assert that each expected function was invoked once
        mock_engine.assert_called_once()
        mock_load_dim.assert_called_once()
        mock_optimize.assert_called_once()
        mock_verify.assert_called_once()


if __name__ == '__main__':
    unittest.main()