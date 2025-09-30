from setuptools import setup, find_packages

setup(
    name="NSMFermions",
    version="0.1.0",
    description="Quantum Machine Learning for Fermions",
    author="Emanuele Costa",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # list your dependencies here
        "numpy",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)