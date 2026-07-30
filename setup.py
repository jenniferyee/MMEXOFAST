from setuptools import find_packages, setup

setup(
    name="mmexofast",
    package_dir={"": "source"},
    packages=find_packages(where="source"),
)
