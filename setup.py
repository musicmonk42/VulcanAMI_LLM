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
        "faker",
        "py-ecc>=6.0.0",  # Required for Groth16 zk-SNARK implementation
    ],
    extras_require={
        "dev": [
            "locust>=2.38.1",  # Load testing (moved from install_requires)
        ]
    },
)