"""
Notebook Bootstrap

Purpose
-------
Provide a zero-config setup for notebooks so users can simply do:

    from src import setup_notebook
    env = setup_notebook()

Defaults
--------
- project_dir : auto-detected project root
- data_dir    : <project_dir>/data
- output_dir  : <data_dir>/output/<notebook_name>/
- cache_dir   : <data_dir>/.cache/<notebook_name>/
- dataset     : "m5"
- use_cache   : True

Behavior
--------
- Outputs and cache live inside the data directory
- dataset="m5" creates <data_dir>/m5
- use_cache=False returns a no-op cache (NullCacheManager),
  so notebook code never needs conditionals
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any
import sys
import warnings

# Optional matplotlib styling (do not crash if matplotlib not installed)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from .helpers import find_project_root, get_notebook_name
from ..cache.cache import CacheManager, ArtifactManager, NullCacheManager


def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_plot_style(style: str) -> str:
    style = (style or "whitegrid").strip().lower()
    if style in {"whitegrid", "seaborn", "seaborn-whitegrid"}:
        return "seaborn-v0_8-whitegrid"
    if style in {"default", "mpl"}:
        return "default"
    return style  # allow direct matplotlib style names


@dataclass(frozen=True)
class NotebookEnvironment:
    PROJECT_DIR: Path
    DATA_DIR: Path
    OUTPUT_DIR: Path
    CACHE_DIR: Path
    NB_NAME: str
    cache: Any
    output: ArtifactManager
    tsf: Any
    dataset: str
    M5_DIR: Optional[Path] = None


def setup_notebook(
    *,
    project_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    dataset: str = "m5",
    use_cache: bool = True,
    show_warnings: bool = False,
    plot_style: str = "whitegrid",
    quiet: bool = False,
) -> NotebookEnvironment:
    """
    Create a clean, consistent notebook environment.

    If you call setup_notebook() with no args, it uses all defaults.

    Optional overrides:
      - project_dir, data_dir, output_dir, cache_dir
      - dataset ("m5" creates data/m5)
      - use_cache (False returns NullCacheManager)
    """
    # --- Project root (default: auto-detect) ---
    root = _as_path(project_dir) or find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # --- Notebook identity ---
    nb_name = get_notebook_name() or Path.cwd().resolve().name

    # --- Resolve directories (defaults inside <project>/data) ---
    data_root = _as_path(data_dir) or (root / "data")
    output_root = _as_path(output_dir) or (data_root / "output")
    cache_root = _as_path(cache_dir) or (data_root / ".cache")

    data_root = _ensure_dir(data_root)
    output_root = _ensure_dir(output_root)
    cache_root = _ensure_dir(cache_root)

    # --- Warnings + plotting style ---
    if not show_warnings:
        warnings.filterwarnings("ignore")
    else:
        warnings.resetwarnings()

    if plt is not None:
        try:
            plt.style.use(_resolve_plot_style(plot_style))
        except Exception:
            pass

    # --- Managers ---
    outputs = ArtifactManager(output_root)
    cache = CacheManager(cache_root) if use_cache else NullCacheManager(cache_root)

    # --- Convenience import: tsforge ---
    try:
        import tsforge as tsf  # type: ignore
    except Exception:
        tsf = None

    # --- Dataset-specific directory ---
    dataset_norm = (dataset or "").strip().lower()
    m5_dir: Optional[Path] = None
    if dataset_norm == "m5":
        m5_dir = _ensure_dir(data_root / "m5")

    if not quiet:
        cache_label = "on" if use_cache else "off"
        print(
            f"✓ Setup complete | Root: {root.name} | Notebook: {nb_name} | "
            f"Data: {data_root} | Cache: {cache_label}"
        )

    return NotebookEnvironment(
        PROJECT_DIR=root,
        DATA_DIR=data_root,
        OUTPUT_DIR=output_root,
        CACHE_DIR=cache_root,
        NB_NAME=nb_name,
        cache=cache,
        output=outputs,
        tsf=tsf,
        dataset=dataset_norm or dataset,
        M5_DIR=m5_dir,
    )
