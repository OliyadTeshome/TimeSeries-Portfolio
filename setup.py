"""
Setup script for TimeSeries-Portfolio package.

This script configures the package for installation and distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

# Development dependencies
dev_requirements = [
    "pytest>=6.0",
    "pytest-cov>=2.10",
    "pytest-xdist>=2.0",
    "pytest-benchmark>=3.4",
    "pytest-mock>=3.6",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
    "mypy>=0.950",
    "pylint>=2.12",
    "bandit>=1.7",
    "safety>=1.10",
    "memory-profiler>=0.60",
    "psutil>=5.8",
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
    "sphinx-autodoc-typehints>=1.12",
    "myst-parser>=0.17",
    "sphinx-copybutton>=0.4",
    "sphinx-tabs>=3.2",
]

setup(
    name="timeseries-portfolio",
    version="1.0.0",
    author="TimeSeries-Portfolio Team",
    author_email="team@timeseries-portfolio.com",
    description="A comprehensive time series analysis and portfolio optimization framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TimeSeries-Portfolio",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TimeSeries-Portfolio/issues",
        "Source": "https://github.com/yourusername/TimeSeries-Portfolio",
        "Documentation": "https://timeseries-portfolio.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "pytest-xdist>=2.0",
            "pytest-benchmark>=3.4",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12",
            "myst-parser>=0.17",
        ],
        "full": dev_requirements,
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    entry_points={
        "console_scripts": [
            "timeseries-portfolio=timeseries_portfolio.cli:main",
        ],
    },
    keywords=[
        "time series",
        "forecasting",
        "portfolio optimization",
        "financial analysis",
        "risk management",
        "backtesting",
        "machine learning",
        "ARIMA",
        "LSTM",
        "finance",
        "quantitative",
    ],
    platforms=["any"],
    license="MIT",
    zip_safe=False,
)
