"""Unit tests for src/scripts/run_quality_report.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the src/scripts directory is on sys.path for imports
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/scripts'))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Mock heavy dependencies before importing the module under test
EXTERNAL_PACKAGES = {
    "polars": MagicMock(),
    "pl": MagicMock(),
}
for pkg_name in EXTERNAL_PACKAGES:
    sys.modules[pkg_name] = EXTERNAL_PACKAGES[pkg_name]

# Now import the module under test
import run_quality_report as rqr


class TestRunQualityReport(unittest.TestCase):
    """Basic sanity checks and flow tests for the quality‑report script."""

    @patch('src.scripts.run_quality_report.pl.read_parquet')
    @patch('src.scripts.run_quality_report.Path')
    @patch('src.scripts.run_quality_report.logger')
    def test_main_successful_flow(self, mock_logger, mock_path_cls, mock_read_parquet):
        """Happy‑path: all file operations succeed and no exception is raised."""
        # ---- Mock Path instances -------------------------------------------------
        mock_path_instance = MagicMock()
        mock_path_instance.__str__.return_value = '/mock/root'
        mock_path_cls.return_value = mock_path_instance

        # ---- Mock Parquet read ----------------------------------------------------
        mock_df = MagicMock()
        mock_df.shape = (100, 5)
        mock_read_parquet.return_value = mock_df

        # Mock load_clean_parquet to return our mock df
        with patch('src.scripts.run_quality_report.load_clean_parquet', return_value=mock_df):
            # Mock generate_report to return a minimal dict
            with patch.object(rqr, 'generate_report', return_value={'total_rows': 100}):
                # Mock save_report to just be called
                with patch.object(rqr, 'save_report'):
                    # Mock log_summary to be called
                    with patch.object(rqr, 'log_summary'):
                        # Run main() – it should execute without raising
                        rqr.main()

        # Verify that the expected logging messages were issued
        mock_logger.info.assert_any_call("=" * 60)
        mock_logger.info.assert_any_call("INICIANDO REPORTE DE CALIDAD DE DATOS")
        mock_logger.info.assert_any_call("REPORTE DE CALIDAD DE DATOS COMPLETADO")
        mock_logger.info.assert_any_call("=" * 60)

    @patch('src.scripts.run_quality_report.pl.read_parquet')
    def test_load_clean_parquet_file_not_found(self, mock_read_parquet):
        """When the parquet file is missing, the function should raise FileNotFoundError."""
        mock_read_parquet.side_effect = FileNotFoundError("Parquet file not found")
        import pathlib
        with patch('src.scripts.run_quality_report.Path'):
            with self.assertRaises(FileNotFoundError):
                rqr.load_clean_parquet(Path('/mock/not_found'))

    @patch('src.scripts.run_quality_report.generate_report')
    def test_generate_report_is_called(self, mock_generate):
        """Ensure generate_report is invoked with the dataframe."""
        # Mock the dataframe that would be passed in
        mock_df = MagicMock()
        with patch('src.scripts.run_quality_report.load_clean_parquet', return_value=mock_df):
            with patch.object(rqr, 'save_report'):  # avoid file write
                with patch.object(rqr, 'log_summary'):  # avoid logging side‑effects
                    rqr.main()
        mock_generate.assert_called_once_with(mock_df)


if __name__ == '__main__':
    unittest.main()