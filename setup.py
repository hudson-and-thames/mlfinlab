# Always prefer setuptools over distutils
from setuptools import setup

setup()

# Create package
# python setup.py bdist_wheel
# python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*  (This is the test repo)
# twine upload dist/*  (This is official repo)
