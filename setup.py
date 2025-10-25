"""Setup configuration for COSC 520 Assignment 2."""

from setuptools import setup, find_packages

setup(
    name="cosc520-a2",
    version="0.1.0",
    description="Advanced probabilistic data structures for COSC 520",
    author="Riley Eaton",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.4.0",
        ],
    },
)
