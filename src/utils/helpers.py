"""
Helper Utilities
================

Small, reusable utility functions for path handling and notebook detection.
"""

from pathlib import Path
from typing import Optional


def find_project_root(marker_files=('.git', 'pyproject.toml')) -> Path:
    """
    Walk up from this module's location until we find a directory with marker files.
    
    Parameters
    ----------
    marker_files : tuple
        Files that indicate project root
        
    Returns
    -------
    Path
        Project root directory
        
    Raises
    ------
    FileNotFoundError
        If project root cannot be found
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError("Could not find project root")


def get_notebook_name() -> Optional[str]:
    """
    Try to detect the current Jupyter notebook name.

    Returns the notebook name without the .ipynb extension, or None if not in a notebook
    or if detection fails. Works with VS Code and JupyterLab.
    
    Returns
    -------
    str or None
        Notebook name (e.g., '1_06_first_contact') or None
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return None

        # Check for VS Code's notebook variable
        if hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
            nb_path = ipython.user_ns['__vsc_ipynb_file__']
            return Path(nb_path).stem

    except Exception:
        pass

    return None


def get_module_from_notebook() -> Optional[str]:
    """
    Extract module identifier (first 4 chars) from notebook name.

    Returns
    -------
    str or None
        Module identifier (e.g., '1_06') or None

    Examples
    --------
    >>> # In notebook "1_06_first_contact.ipynb"
    >>> get_module_from_notebook()
    '1_06'
    """
    nb_name = get_notebook_name()
    return nb_name[:4] if nb_name and len(nb_name) >= 4 else None


def get_notebook_path() -> Optional[Path]:
    """
    Get the full path to the current notebook.

    Returns
    -------
    Path or None
        Full path to notebook file, or None if not in a notebook
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return None

        if hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
            return Path(ipython.user_ns['__vsc_ipynb_file__'])

    except Exception:
        pass

    return None


def get_artifact_subfolder() -> Optional[str]:
    """
    Derive artifact subfolder from notebook's parent directory.

    Strips 'module_' prefix from the parent directory name.

    Returns
    -------
    str or None
        Subfolder name (e.g., '01_foundations') or None

    Examples
    --------
    >>> # In notebook at notebooks/module_01_foundations/1_06_first_contact.ipynb
    >>> get_artifact_subfolder()
    '01_foundations'
    """
    nb_path = get_notebook_path()
    if nb_path is None:
        return None

    parent_name = nb_path.parent.name
    # Strip 'module_' prefix if present
    if parent_name.startswith('module_'):
        return parent_name[7:]  # len('module_') == 7
    return parent_name


__all__ = [
    'find_project_root',
    'get_notebook_name',
    'get_notebook_path',
    'get_module_from_notebook',
    'get_artifact_subfolder',
]
