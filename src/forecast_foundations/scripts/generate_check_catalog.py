#!/usr/bin/env python3
"""
Print the 5Q check catalog (stable keys) for a module.

Usage:
  python scripts/generate_check_catalog.py 1.06
  python scripts/generate_check_catalog.py 1.08
"""
from __future__ import annotations
import sys
from pprint import pprint

from reports import list_checks, MODULE_TITLES

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_check_catalog.py <module_id>")
        sys.exit(1)

    module = sys.argv[1]
    title = MODULE_TITLES.get(module, "")
    print(f"{module} · {title}".strip())
    print("-" * 60)
    catalog = list_checks(module)
    if not catalog:
        print("No catalog found.")
        return
    pprint(catalog, sort_dicts=False)

if __name__ == "__main__":
    main()
