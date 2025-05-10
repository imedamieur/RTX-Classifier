from setuptools import setup, find_packages

setup(
    name="rtx-classifier",
    version="0.1.0",
    description="Reverse-Transaction Classifier for AAOIFI Standards",
    author="RTX Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)