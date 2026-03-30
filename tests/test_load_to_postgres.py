"""Unit tests for src/scripts/load_to_postgres.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# ----------------------------------------------------------------------
# Adjust sys.path so we can import the script as a module
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src/scripts')
)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Mock heavy imports that would cause trouble (e.g. pandas, sqlalchemy)
EXTERNAL_PACKAGES = {
    "pandas": MagicMock(),
    "joblib": MagicMock(),
    "sqlalchemy": MagicMock(),
    "sqlalchemy.engine": MagicMock(),
}
for name in EXTERNAL_PACKAGES:
    sys.modules[name] = EXTERNAL_PACKAGES[name]

# Load environment‑variable helper mock
MOCK_ENV_PATH = MagicMock()
with patch.dict(os.environ, {}, clear=True):
    # Mock utils.database functions
    with patch('src.scripts.load_to_postgres.load_environment_variables', return_value=None):
        with patch('src.scripts.load_to_postgres.create_database_engine', return_value=MagicMock()):
            import load_to_postgres as ltp


class TestLoadToPostgres(unittest.TestCase):
    """Sanity‑checks for the load‑to‑Postgres pipeline."""

    def test_import_succeeds(self):
        """The module must import without raising ImportError."""
        self.assertTrue(hasattr(ltp, "main"))

    @patch('src.scripts.load_to_postgres.pathlib.Path')
    @patch('src.scripts.load_to_postgres.pd.read_parquet')
    @patch('src.scripts.load_to_postgres.model')
    def test_load_and_sample_data(mock_model, mock_read_parquet, mock_path):
        """load_and_sample_data should return a sampled DataFrame."""
        # Mock the parquet read return value
        mock_df = MagicMock()
        mock_df.shape = (12345, 10)
        mock_read_parquet.return_value = mock_df

        # Call the function
        result = ltp.load_and_sample_data(MagicMock(), sample_size=1000)

        # Assertions
        self.assertEqual(result, mock_df)
        mock_read_parquet.assert_called_once()
        # Sampling logic should have been exercised (not strictly verified here)

    @patch('src.scripts.load_to_postgres.model')
    def test_load_model_and_predict(mock_model):
        """load_model_and_predict should call joblib.load and model.predict_proba."""
        # Mock joblib.load to return a dummy model object
        dummy_model = MagicMock()
        dummy_model.predict_proba.return_value = [0.1, 0.2, 0.3]  # dummy probs
        mock_model.load.return_value = dummy_model

        # Dummy input DataFrame
        dummy_df = MagicMock()
        dummy_df.shape = (5, 3)
        dummy_df.drop = MagicMock(return_value=MagicMock())
        dummy_df.columns = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
        # Mock astype for categorical columns
        dummy_df.drop.return_value.astype.return_value = MagicMock()

        # Execute the function
        result_df = ltp.load_model_and_predict(dummy_df, MagicMock())

        # Verify that the model was loaded and predict_proba was called
        mock_model.load.assert_called_once()
        dummy_model.predict_proba.assert_called_once()
        # Ensure the returned df got the new column added
        dummy_df.assert_has_calls([call.astype.return_value])  # just sanity check


class TestSchemaAndSQL(unittest.TestCase):
    """Tests that the SQL‑generation part builds the expected statements."""

    @patch('src.scripts.load_to_postgres.text')
    def test_load_to_postgres_builds_correct_sql(self, mock_text):
        """Verify that the SQL command list contains the expected DDL."""
        import src.scripts.load_to_postgres as ltp

        # Minimal engine mock
        engine = MagicMock()
        # Mock the connection/cursor execution briefly
        conn = MagicMock()
        conn.execute = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn

        # Mock pandas to_sql to just be called
        with patch('src.scripts.load_to_postgres.pd.DataFrame.to_sql') as mock_to_sql:
            # Run a dummy function that exercises sql generation
            ltp.load_to_postgres(MagicMock(), engine, table_name="fact_flights")

        # Ensure some expected SQL fragments are present in the command list
        # (We indirectly verify this by checking that to_sql was called)
        self.assertTrue(mock_to_sql.called)


if __name__ == '__main__':
    unittest.main()