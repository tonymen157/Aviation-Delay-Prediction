"""Utility functions shared across ETL pipeline scripts."""

import sys
from pathlib import Path
from utils.logging_config import setup_logger

# Create a module-level logger for scripts that use this module
logger = setup_logger(__name__)


def add_project_root_to_path() -> None:
    """
    Insert the project root directory at the beginning of sys.path
    so that top‑level packages (e.g. `utils`, `models`) can be imported
    consistently from any script location.
    """
    project_root = Path(__file__).resolve().parents[2]  # repo root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def run_quality_report() -> None:
    """
    Execute the quality‑report script (`src/scripts/run_quality_report.py`)
    from within another script.  This centralises the call so every
    pipeline step can simply invoke ``run_quality_report()`` instead of
    duplicating ``subprocess.run`` boilerplate.
    """
    from subprocess import run

    # Resolve the script path relative to this module
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_quality_report.py"
    if not script_path.exists():
        logger.warning("Quality‑report script not found at %s", script_path)
        return

    # Run it synchronously, propagating stdout/stderr to the caller's logs
    try:
        result = run([sys.executable, str(script_path)], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Quality‑report generated successfully.")
            logger.debug(result.stdout)
        else:
            logger.warning("Error executing quality‑report: %s", result.stderr)
    except Exception as exc:  # pragma: no cover – defensive
        logger.error("Exception while running quality‑report: %s", exc)