"""
Internal utility helpers for notebooks and modules.

These are not meant to be imported directly by beginners.
"""
from .helpers import (
    find_project_root,
    get_notebook_name,
    get_notebook_path,
    get_module_from_notebook,
    get_artifact_subfolder,
    plot_ld6_vs_sb,
    ld6_vs_sb_summary,
)

from .bootstrap import setup_notebook, NotebookEnvironment

__all__ = [
    "find_project_root",
    "get_notebook_name",
    "get_notebook_path",
    "get_module_from_notebook",
    "get_artifact_subfolder",
    "setup_notebook",
    "NotebookEnvironment",
    "plot_ld6_vs_sb",
    "ld6_vs_sb_summary",
    "plot_ld6_vs_sb",
]
