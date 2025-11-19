from setuptools import setup, find_packages

setup(
    name="graphix",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["graph_compiler", "load_test"],  # Root-level modules
    install_requires=[
        "networkx",
        "numpy",
        "llvmlite",
        "locust",
        "faker",
    ],
)