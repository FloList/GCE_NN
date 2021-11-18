import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gce-nn",
    version="0.0.1",
    author="Florian List",
    author_email="florian.list@univie.ac.at",
    description="Galactic Center Excess Neural Network package",
    long_description=long_description,   
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"GCE": "GCE"},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    long_description_content_type="text/markdown"
)
