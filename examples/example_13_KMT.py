"""
Show an example for implementing detection thresholds for anomaly detection, using KMTNet data as an example.
"""

import glob
import os.path

import exozippy

af_limits = {"tol_zero": 120.0, "tol_flat": 35.0, "tol_zero_alt": 75.0}

# Light curve without a planet
pspl_files = glob.glob(
    os.path.join(exozippy.MULENS_DATA_PATH, "KMTPSPLEvent", "*.dat")
)
results_pspl = exozippy.mmexofast.fit(
    files=[pspl_files], print_results=True, af_tol=af_limits
)

# Light curve with a planet
planet_files = glob.glob(
    os.path.join(exozippy.MULENS_DATA_PATH, "KMTPlanetEvent", "*.dat")
)
results_planet = exozippy.mmexofast.fit(
    files=[planet_files], print_results=True, af_tol=af_limits
)
