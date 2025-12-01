# thesis_code/setup.py
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="TFM",
    version="0.1.0",
    author="a0",
    description="TFM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    license="MIT",

    package_dir={"": "src"},
    packages=find_packages("src"),

    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "opencv-python",
    ],
    include_package_data=True,
)