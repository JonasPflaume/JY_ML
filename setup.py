import sys
from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "jymlctr"
DESCRIPTION = "jiayun's machine learning & control packages"
URL = ""
EMAIL = "li.jiayun@outlook.com"
AUTHOR = "Jiayun Li"
REQUIRES_PYTHON = ">=3.6.9"
VERSION = "0.0.1"
#torch==1.7.1
REQUIRED = [
    "numpy==1.19.5",
    "scipy==1.5.4",
    "matplotlib==3.3.3", 
    "torch",
    "gym==0.21.0",
    "sklearn",
    "stable-baselines3==1.0",
    "gpytorch"
]

setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["jylearn", "jycontrol"],
    install_requires=REQUIRED,
    include_package_data=True,
    license=""
)
