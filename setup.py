from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Machine Learning Labs repo is designed to work with Python 3.6" \
    + " and greater. Please install it before proceeding."

__version__ = '0.0.0'

setup(
    name='machine-learning-labs',
    py_modules=['machine-learning-labs'],
    version=__version__,
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy'
    ],
    description="Excercises in machine learning.",
    author="Dylan Holmes",
)
