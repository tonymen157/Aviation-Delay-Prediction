"""Unit tests for src/scripts/download_data.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Adjust import path so we can import the script as a module
SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts')
)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from src.scripts.download_data import download_and_organize_data

class TestDownloadData(unittest.TestCase):
    """Test the download_and_organize_data function."""

    @patch('kagglehub.dataset_download')
    @patch('shutil.copy2')
    def test_successful_download(self, mock_copy2, mock_dataset_download):
        """Mock a successful download and verify file copying occurs."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            cache_path.mkdir()

            # Create mock files in cache
            file1 = cache_path / "file1.csv"
            file1.write_text("test data 1")
            file2 = cache_path / "file2.csv"
            file2.write_text("test data 2")

            mock_dataset_download.return_value = str(cache_path)

            # Mock Path for the raw data directory
            raw_data_dir = Path(temp_dir) / "data" / "raw"

            # Execute
            download_and_organize_data(raw_data_dir=raw_data_dir)

            # Verify that copy2 was called for each file
            self.assertEqual(mock_copy2.call_count, 2)
            mock_copy2.assert_any_call(file1, raw_data_dir / "file1.csv")
            mock_copy2.assert_any_call(file2, raw_data_dir / "file2.csv")

    @patch('kagglehub.dataset_download')
    def test_download_failure(self, mock_dataset_download):
        """Test error handling when dataset_download raises."""
        mock_dataset_download.side_effect = RuntimeError("Network error")

        with self.assertRaises(SystemExit) as cm:
            download_and_organize_data()
        self.assertEqual(cm.exception.code, 1)

if __name__ == '__main__':
    unittest.main()