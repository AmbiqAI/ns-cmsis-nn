"""
CMSIS-NN Tools Package

A comprehensive toolkit for CMSIS-NN testing, model generation, and FVP simulation.
"""

__version__ = "0.1.0"
__author__ = "Helia-Core Team"

from cmsis_nn_tools.core.config import Config
from cmsis_nn_tools.core.pipeline import FullTestPipeline

__all__ = [
    "Config",
    "FullTestPipeline",
]
