from setuptools import setup, find_packages

#build with: python setup.py sdist bdist_wheel

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ccbspline",
    version="0.1.0",
    author="Karl Haislmaier",
    author_email="karlh.academic@gmail.com",
    description="A numba-accelerated Cubic Cardinal B-Spline drop-in replacement for scipy.interplate.CubicSpline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khslmr/numba-cardinal-B-splines",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ]
)
