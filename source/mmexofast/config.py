"""Configuration settings, including path definitions for the module and data directory."""

# Import paths
from os import path

MODULE_PATH = path.abspath(__file__)
for i in range(3):
    MODULE_PATH = path.dirname(MODULE_PATH)

# Data shipped inside the package: the Spitzer ephemerides that the built-in
# Spitzer observatory needs at runtime, plus the sample events and unit test
# fixtures (868 KB in total). Because this lives in the package it resolves
# identically in a source checkout and an installed package, so the unit
# tests run either way.
PACKAGE_DATA_PATH = path.join(path.dirname(path.abspath(__file__)), "data")

# DATA_PATH is the packaged data root. It is always present.
DATA_PATH = PACKAGE_DATA_PATH

# The 2018 Data Challenge (data/2018DataChallenge) is the one dataset that is
# not shipped: 132 MB of light curves, over PyPI's 100 MB per-file limit, and
# the upstream repository carries no license, so it is not ours to
# redistribute. It exists only in a source checkout, hence None for an
# installed package. See observatories.get_dc18_ephemerides for how the one
# file needed at runtime is obtained without redistributing it.
_DC18_PARENT = path.join(MODULE_PATH, "data")
SAMPLE_DATA_PATH = _DC18_PARENT if path.isdir(_DC18_PARENT) else None
