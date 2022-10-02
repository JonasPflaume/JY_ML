import sys
from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "jylearn"
DESCRIPTION = "jiayun's machine learning package"
URL = ""
EMAIL = "li.jiayun@outlook.com"
AUTHOR = "Jiayun Li"
REQUIRES_PYTHON = ">=3.6.9"
VERSION = "0.0.1"

REQUIRED = [
    "numpy==1.19.5", "scipy", "matplotlib", "torch==1.7.1"
]

setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["jylearn"],
    install_requires=REQUIRED,
    include_package_data=True,
    license=""
)
