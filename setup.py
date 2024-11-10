import io
import os
from setuptools import find_packages, setup


def read_requirements(file_name="requirements.txt"):
    """Read requirements from a file and return as a list."""
    with open(file_name) as f:
        return f.read().splitlines()


def read(file_name):
    """Read a text file and return the content as a string."""
    with io.open(
            os.path.join(os.path.dirname(__file__), file_name), encoding="utf-8"
    ) as f:
        return f.read()


# Setup configuration
setup(
    name="hextech",
    version="0.0.1",
    description="A brief description of the hextech project",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",  # Update license if needed
        "Operating System :: OS Independent",
    ],
    package_data={"hextech": ["py.typed", "logging.conf"], "": ["*.jinja2"]},
    packages=find_packages(),
    install_requires=read_requirements(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
)
