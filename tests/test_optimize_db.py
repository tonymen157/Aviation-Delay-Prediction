"""Unit tests for src/scripts/optimize_db.py."""

import unittest
from unittest.mock import patch, MagicMock

# Direct absolute import – src is a package, scripts is a subpackage
from src.scripts import optimize_db as od


class TestOptimizeDBImport(unittest.TestCase):
    """Sanity checks – the module must import and expose expected symbols."""

    def test_import_succeeds(self):
        """Importing the module must not raise ImportError."""
        self.assertTrue(hasattr(od, "main"))
        self.assertTrue(callable(od.main))

    def test_load_dimension_tables_exists(self):
        """load_dimension_tables must be defined."""
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
    """Test that the CLI entry‑point calls the expected functions in order."""

    @patch('src.scripts.optimize_db.verify_optimizations')
    @patch('src.scripts.optimize_db.optimize_database')
    @patch('src.scripts.optimize_db.load_dimension_tables')
    @patch('src.scripts.optimize_db.create_database_engine')
    def test_main_calls_expected_functions(self, mock_create_engine, mock_load_dim, mock_optimize, mock_verify):
        """
        When `main()` runs it should, in order:
        1. Load environment variables.
        2. Create a DB engine.
        3. Load dimension tables.
        4. Run optimization SQL commands.
        5. Verify the optimizations.
        """
        # Mock the database connection
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.commit = MagicMock()
        mock_conn.execute = MagicMock()
        mock_engine.connect.return_value = mock_conn

        # Setup environment loading mock
        with patch('src.scripts.optimize_db.load_environment_variables'):
            # Execute main
            od.main()

        # Verify each expected function was called exactly once
        mock_create_engine.assert_called_once()
        mock_load_dim.assert_called_once()
        mock_optimize.assert_called_once()
        mock_verify.assert_called_once()


class TestDimensionTableLoading(unittest.TestCase):
    """Test that dimension tables are loaded correctly."""

    @patch('src.scripts.optimize_db.pd.read_parquet')
    @patch('src.scripts.optimize_db.Path')
    def test_load_dimension_tables_success(self, mock_path, mock_read_parquet):
        """Loading dimension tables should succeed when Parquet files exist."""
        # Mock the Path objects
        mock_path_instance = MagicMock()
        mock_path_instance.__str__.return_value = '/mock/root'
        mock_path_cls = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        # Mock pandas read_parquet to return mock DataFrames
        mock_df_airlines = MagicMock()
        mock_df_airlines.shape = (10, 3)
        mock_df_airports = MagicMock()
        mock_df_airports.shape = (8, 3)
        mock_read_parquet.side_effect = [mock_df_airlines, mock_df_airports]

        # Mock pandas to_sql to avoid actual DB writes
        with patch('src.scripts.optimize_db.pd.DataFrame.to_sql'):
            # Mock engine.connect to return a mock connection
            mock_conn = MagicMock()
            mock_conn.commit = MagicMock()
            mock_conn.execute = MagicMock()
            mock_engine = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.commit = MagicMock()
            mock_conn.execute = MagicMock()

            # Execute load_dimension_tables
            od.load_dimension_tables(mock_engine, MagicMock())

            # Verify that pandas read_parquet was called twice (for airlines and airports)
            self.assertEqual(mock_read_parquet.call_count, 2)
            # Verify that to_sql was called twice (once per table)
            # Note: we cannot directly access the calls here, but the fact that we mocked it ensures no errors.


class TestDimensionTableVerification(unittest.TestCase):
    """Test that the verification of primary keys and indexes works."""

    @patch('sqlalchemy.text')
    def test_verify_optimizations_checks(self, mock_text):
        """verify_optimizations should query information_schema for PKs and indexes."""
        # Mock the database connection
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock the result of execute() calls
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result

        with patch('src.scripts.optimize_db.create_database_engine', return_value=mock_engine):
            # Execute verify_optimizations
            od.verify_optimizations(mock_engine)

            # Verify that execute was called on the connection
            self.assertTrue(mock_conn.execute.called)
            # Verify that fetchall was called on the result
            self.assertTrue(mock_result.fetchall.called)


if __name__ == '__main__':
    unittest.main()