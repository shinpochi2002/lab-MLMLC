#!/usr/bin/env python3
"""Independent Phase 1/2 entry point; see this directory's README.md."""
import sys
sys.dont_write_bytecode = True
from pipeline import main

if __name__ == "__main__":
    main('summarize')
