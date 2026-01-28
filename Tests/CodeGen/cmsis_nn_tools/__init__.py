"""
Helia-Core Tester Package

A comprehensive toolkit for Helia-Core kernels testing, model generation, and FVP simulation.
"""

__version__ = "1.0.0"
__author__ = "Helia-Core Tester Team"

from .core.pipeline import FullTestPipeline

__all__ = [
    "FullTestPipeline",
]
