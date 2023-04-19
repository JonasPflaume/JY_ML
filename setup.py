from setuptools import setup

# Package Metadata
NAME = "jymlctr"
DESCRIPTION = "jiayun's machine learning & control packages. \
                aslearn:    machine learning packages\
                asctr:      control packages\
                asopt:      optimization algorithms packages"
URL = ""
EMAIL = "li.jiayun@outlook.com"
AUTHOR = "Jiayun Li"
REQUIRES_PYTHON = ">=3.9"
VERSION = "0.0.1"
#torch==1.7.1
REQUIRED = [
    "numpy",
    "scipy",
    "matplotlib", 
    "torch",
    "gym==0.21.0",
    "stable-baselines3",
    "gpytorch",
    "casadi",
    "numba"
]

setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["aslearn", "asctr", "asopt"],
    install_requires=REQUIRED,
    include_package_data=True,
    license=""
)
