from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="RandomForest",
    version="0.1",
    packages=find_packages(),
 
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["docutils>=0.3",
    									"numpy>=1.19.0",
    									"pandas>=1.0.5",
    									"pytest>=5.4.3"],


    # metadata to display on PyPI
    author="Mike Harmon",
    author_email="mdh266@gmail.com",

    python_requires='>=3.6'
)
