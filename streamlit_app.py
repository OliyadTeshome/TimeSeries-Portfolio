"""
Streamlit entrypoint that loads the dashboard as a package module.

This avoids "attempted relative import with no known parent package" errors
by ensuring the project root is on sys.path and importing `src.dashboard` as
part of the `src` package.
"""

import sys
from pathlib import Path


def _ensure_project_root_on_syspath() -> None:
    """Prepend the repository root to sys.path if missing."""
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def main() -> None:
    _ensure_project_root_on_syspath()
    # Import after adjusting sys.path so relative imports inside src work
    from src.dashboard import main as dashboard_main  # noqa: WPS433 (local import)

    dashboard_main()


if __name__ == "__main__":
    main()


