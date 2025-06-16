"""
OML (Open, Monetizable, and Loyal AI) - Top-down Architecture

A comprehensive framework for AI model fingerprinting, utility evaluation, and adversarial testing.
"""

__version__ = "2.0.0"
__author__ = "Sentient Research"

from . import adversary
from . import utility
from . import verification
from . import common

__all__ = ["adversary", "utility", "verification", "common"] 