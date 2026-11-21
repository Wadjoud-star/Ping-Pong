#!/usr/bin/env python3
"""Lanceur pratique depuis la racine du projet."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [str(ROOT / "src" / "main.py"), *sys.argv[1:]]
runpy.run_path(str(ROOT / "src" / "main.py"), run_name="__main__")
