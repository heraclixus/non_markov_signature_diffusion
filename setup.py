from setuptools import setup, find_packages

setup(
    name="nmsd",
    version="0.1.0",
    description="Non-Markov Signature Diffusion",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision",
        "numpy",
        "pyyaml",
        "tqdm",
        "matplotlib",
        # "pysiglib"  # Assuming this is installed separately or manually
    ],
)

