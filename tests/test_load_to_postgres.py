"""Unit tests for src/scripts/load_to_postgres.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

# ----------------------------------------------------------------------
# Adjust sys.path so we can import the script as a module
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src/scripts')
)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Mock heavy dependencies that would cause import errors
EXTERNAL_PACKAGES = {
    "pandas": MagicMock(),
    "joblib": MagicMock(),
    "sqlalchemy": MagicMock(),
    "sqlalchemy.engine": MagicMock(),
}
for name in EXTERNAL_PACKAGES:
    sys.modules[name] = EXTERNAL_PACKAGES[name]

# Import the module *after* patching the parts that would cause ImportError
# First, patch the environment‑variable loader (it reads .env files)
with patch('src.scripts.load_to_postgres.load_environment_variables', return_value=None):
    # Now we can safely import the target module
    from src.scripts import load_to_postgres as ltp


class TestLoadToPostgresImport(unittest.TestCase):
    """Sanity checks – the module must import and expose expected symbols."""

    def test_import_succeeds(self):
        """Importing the module must not raise ImportError."""
        self.assertTrue(hasattr(ltp, "main"))
        self.assertTrue(callable(ltp.main))

    def test_main_exists(self):
        """The CLI entry‑point must be defined."""
        self.assertTrue(hasattr(ltp, "main"))
        self.assertTrue(callable(ltp.main))


class TestLoadFunctions(unittest.TestCase):
    """Basic tests for the individual helper functions."""

    @patch('src.scripts.load_to_postgres.Path')
    @patch('src.scripts.load_to_postgres.pd.read_parquet')
    @patch('src.scripts.load_to_postgres.logger')
    def test_load_and_sample_data_returns_df(self, mock_logger, mock_read_parquet, mock_path):
        """load_and_sample_data should return the DataFrame read from parquet."""
        mock_df = MagicMock()
        mock_df.shape = (12345, 10)
        mock_read_parquet.return_value = mock_df

        mock_parquet_path = MagicMock()
        mock_parquet_path.__str__.return_value = "/fake/path.parquet"
        result = ltp.load_and_sample_data(mock_parquet_path, sample_size=1000)
        self.assertEqual(result, mock_df)
        mock_read_parquet.assert_called_once()

    @patch('src.scripts.load_to_postgres.pd.read_parquet')
    def test_load_and_sample_data_respects_sample_size(self, mock_read_parquet):
        """When the dataset is larger than sample_size, the sample size is respected."""
        from unittest.mock import PropertyMock
        # Create a mock DataFrame
        mock_df = MagicMock()
        # Make shape behave like a real tuple
        type(mock_df).shape = PropertyMock(return_value=(5000, 8))
        # When sample is called, return another mock that has shape (100, 8)
        mock_sample_result = MagicMock()
        type(mock_sample_result).shape = PropertyMock(return_value=(100, 8))
        mock_df.sample.return_value = mock_sample_result

        mock_read_parquet.return_value = mock_df

        # Use a real string path for clarity
        result = ltp.load_and_sample_data("/fake/path.parquet", sample_size=100)
        # The function should return the sampled result
        self.assertEqual(result, mock_sample_result)
        mock_read_parquet.assert_called_once_with("/fake/path.parquet")
        # Verify that sample was called with the correct arguments
        mock_df.sample.assert_called_once_with(n=100, random_state=42)

    @patch('src.scripts.load_to_postgres.model')
    def test_load_model_and_predict_calls_joblib_and_creates_column(mock_model):
        """load_model_and_predict must load the model and add DELAY_PROBABILITY column."""
        # Mock joblib.load to return a dummy model
        dummy_model = MagicMock()
        dummy_model.predict_proba.return_value = [0.12, 0.34, 0.55]
        mock_model.load.return_value = dummy_model

        # Dummy input DataFrame
        dummy_df = MagicMock()
        dummy_df.shape = (5, 3)
        dummy_df.drop = MagicMock(return_value=MagicMock())
        dummy_df.columns = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

        # Ensure categorical columns are cast to category type
        dummy_df.drop.return_value.astype.return_value = MagicMock()

        result_df = ltp.load_model_and_predict(dummy_df, MagicMock())
        # Verify that the model was loaded
        mock_model.load.assert_called_once()
        # Verify predict_proba was called
        dummy_model.predict_proba.assert_called_once()
        # Verify new column was added
        self.assertIn('DELAY_PROBABILITY', result_df.columns)

    @patch('src.scripts.load_to_postgres.pd.read_parquet')
    @patch('src.scripts.load_to_postgres.engine')
    def test_load_to_postgres_calls_to_sql_and_passes_correct_params(mock_engine, mock_read_parquet):
        """Ensure that to_sql receives the expected parameters."""
        # Mock the DataFrame to be written
        dummy_df = MagicMock()
        dummy_df.shape = (1000, 5)

        # Mock engine.connect to return a mock connection
        mock_conn = MagicMock()
        mock_conn.commit = MagicMock()
        mock_conn.execute = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock pandas to_sql to avoid actual DB interaction
        with patch('src.scripts.load_to_postgres.pd.DataFrame.to_sql') as mock_to_sql:
            # Execute the function (it will call to_sql internally)
            dummy_engine = mock_engine.return_value
            dummy_df = dummy_df  # noqa: F401
            ltp.load_to_postgres(dummy_df, dummy_engine, table_name='fact_flights')

            # Verify that to_sql was called with the expected arguments
            self.assertTrue(mock_to_sql.called)
            called_args, _ = mock_to_sql.call_args
            called_kwargs = called_args[1]  # kwargs passed to to_sql
            self.assertEqual(called_kwargs.get('if_exists'), 'replace')
            self.assertEqual(called_kwargs.get('index'), False)
            self.assertEqual(called_kwargs.get('chunksize'), 10_000)
            self.assertEqual(called_kwargs.get('method'), 'multi')
            self.assertEqual(called_kwargs.get('name'), 'fact_flights')

    @patch('src.scripts.load_to_postgres.pd.read_parquet')
    def test_load_to_postgres_missing_parquet_raises(mock_read_parquet):
        """If the parquet file does not exist, the function should raise FileNotFoundError."""
        mock_read_parquet.side_effect = FileNotFoundError("Parquet file not found")
        with self.assertRaises(FileNotFoundError):
            ltp.load_and_sample_data(MagicMock(), sample_size=100)

    @patch('src.scripts.load_to_postgres.model')
    def test_load_model_and_predict_handles_missing_model(mock_model):
        """If the model file cannot be loaded, the function should propagate the exception."""
        mock_model.load.side_effect = FileNotFoundError("Model file not found")
        with self.assertRaises(FileNotFoundError):
            ltp.load_model_and_predict(MagicMock(), MagicMock())

    @patch('src.scripts.load_to_postgres.pd.DataFrame.to_sql')
    @patch('src.scripts.load_to_postgres.engine.connect')
    def test_load_to_postgres_error_handling_on_sql_execution(mock_conn, mock_to_sql):
        """If an SQL error occurs during to_sql, the transaction should be rolled back."""
        # Setup mocks
        mock_conn.commit = MagicMock()
        mock_conn.execute.side_effect = Exception("SQL error")
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        with patch('src.scripts.load_to_postgres.create_database_engine', return_value=mock_engine):
            with self.assertRaises(Exception):
                ltp.load_to_postgres(MagicMock(), mock_engine, table_name='fact_flights')

            # Verify rollback was called
            self.assertTrue(mock_conn.rollback.called)

    @patch('src.scripts.load_to_postgres.logger')
    def test_main_logs_start_and_end_messages(self, mock_logger):
        """The main function should emit start and completion log messages."""
        with patch('src.scripts.load_to_postgres.load_environment_variables'):
            with patch('src.scripts.load_to_postgres.create_database_engine'):
                with patch('src.scripts.load_to_postgres.load_and_sample_data') as mock_sample:
                    with self.assertRaises(SystemExit):  # SystemExit is raised after main finishes
                        ltp.main()
                # Verify that INFO‑level messages for start and end were logged
                self.assertTrue(mock_logger.info.called)
                # Look for calls containing the start and end phrases
                start_calls = [call for call in mock_logger.info.call_args_list
                               if 'INICIANDO' in call[0][0]]
                end_calls = [call for call in mock_logger.info.call_args_list
                               if 'COMPLETADO' in call[0][0]]
                self.assertTrue(len(start_calls) > 0)
                self.assertTrue(len(end_calls) > 0)

    @patch('src.scripts.load_to_postgres.logger')
    def test_main_exits_with_systemexit_on_error(self, mock_logger):
        """When an unhandled exception occurs, main() should exit with code 1."""
        with patch('src.scripts.load_to_postgres.load_environment_variables'):
            with patch('src.scripts.load_to_postgres.create_database_engine', side_effect=Exception("Fatal error")):
                with self.assertRaises(SystemExit) as cm:
                    ltp.main()
                self.assertEqual(cm.exception.code, 1)


class TestSQLCommandGeneration(unittest.TestCase):
    """Verify that the generated SQL command list is correct."""

    @patch('src.scripts.load_to_postgres.text')
    def test_sql_commands_are_correct(self, mock_text):
        """Check that the list of SQL commands contains the expected statements."""
        # Minimal engine mock
        mock_conn = MagicMock()
        mock_conn.commit = MagicMock()
        mock_conn.execute = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock pandas to_sql to avoid actual DB interaction
        with patch('src.scripts.load_to_postgres.pd.DataFrame.to_sql'):
            # Execute a dummy flow that would generate the SQL commands
            from src.scripts import load_to_postgres as ltp
            # Mock environment loading
            with patch('src.scripts.load_to_postgres.load_environment_variables'):
                with patch('src.scripts.load_to_postgres.create_database_engine', return_value=mock_engine):
                    # Call a function that runs the optimization (it will execute the commands)
                    # We directly test the function that holds the SQL list
                    sql_commands = [
                        'ALTER TABLE dim_airlines ADD PRIMARY KEY ("IATA_CODE");',
                        'ALTER TABLE dim_airports ADD PRIMARY KEY ("IATA_CODE");',
                        'CREATE INDEX IF NOT EXISTS idx_fact_month ON fact_flights ("MONTH");',
                        'CREATE INDEX IF NOT EXISTS idx_fact_airline ON fact_flights ("AIRLINE");',
                        'CREATE INDEX IF NOT EXISTS idx_fact_origin ON fact_flights ("ORIGIN_AIRPORT");',
                        'CREATE INDEX IF NOT EXISTS idx_fact_target ON fact_flights ("TARGET_IS_DELAYED");',
                        'CREATE INDEX IF NOT EXISTS idx_fact_month_airline ON fact_flights ("MONTH", "AIRLINE");',
                    ]
                    # Compare the generated list (the real code builds this list internally)
                    self.assertListEqual(sql_commands, ltp.sql_commands)


if __name__ == '__main__':
    unittest.main()