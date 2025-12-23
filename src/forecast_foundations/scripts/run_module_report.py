#!/usr/bin/env python3
"""
Run a ModuleReport and print it.

Usage:
  python scripts/run_module_report.py 1.06 path/to/input.parquet path/to/output.parquet
  python scripts/run_module_report.py 1.08 path/to/input.parquet path/to/output.parquet

Notes:
- expects columns: unique_id, ds, y (or override via env/kwargs in your notebook)
"""
from __future__ import annotations
import sys
import pandas as pd

from reports import ModuleReport

def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/run_module_report.py <module_id> <input_path> <output_path>")
        sys.exit(1)

    module, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    input_df = pd.read_parquet(in_path) if in_path.endswith(".parquet") else pd.read_csv(in_path)
    output_df = pd.read_parquet(out_path) if out_path.endswith(".parquet") else pd.read_csv(out_path)

    report = ModuleReport(module, input_df, output_df)
    report.display()

if __name__ == "__main__":
    main()
