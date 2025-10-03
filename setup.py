"""Setup script for Reggie"""

from setuptools import setup, find_packages

setup(
    name="reggie",
    version="0.1.0",
    description="AI Agent for exploring Regulations.gov",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "reggie=reggie.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "reggie": ["db/*.sql"],
    },
)
