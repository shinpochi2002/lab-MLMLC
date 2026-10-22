#!/usr/bin/env python3
"""Classifier-local partition entry point; no model training."""
import sys
sys.dont_write_bytecode = True
from classifier_pipeline import main

if __name__ == "__main__":
    main('create')
