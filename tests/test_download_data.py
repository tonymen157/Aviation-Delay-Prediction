"""Unit tests for src/scripts/download_data.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src/scripts directory to the import path
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '../../src/scripts')
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from download_data import download_and_organize_data


class TestDownloadData(unittest.TestCase):
    """Test the download_and_organize_data function."""

    @patch('kagglehub.dataset_download')
    def test_successful_download(self, mock_dataset_download):
        """Mock a successful download and verify file copying occurs."""
        # Mock the path returned by dataset_download
        mock_path = Path('/tmp/kagglehub')
        mock_dataset_download.return_value = mock_path

        # Mock the filesystem operations
        mock_copy2 = MagicMock()
        with patch('shutil.copy2', mock_copy2), \
             patch.object(Path, 'iterdir', return_value=[Path('/tmp/kagglehub/file1.csv'), Path('/tmp/kagglehub/file2.csv')]):

            # Run the function (it will try to copy files)
            download_and_organize_data()

            # Assert that copy2 was called for each file
            self.assertTrue(mock_copy2.called)
            self.assertEqual(mock_copy2.call_count, 2)

    @patch('kagglehub.dataset_download')
    def test_download_failure(self, mock_dataset_download):
        """Test error handling when dataset_download raises."""
        mock_dataset_download.side_effect = RuntimeError("Network error")

        with self.assertRaises(SystemExit) as cm:
            download_and_organize_data()
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()