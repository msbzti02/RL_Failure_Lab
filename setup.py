from setuptools import setup, find_packages

setup(
    name="rl-failure-lab",
    version="0.1.0",
    description="A modular RL experimentation framework for studying failure modes in sequential decision-making",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tensorboard>=2.14.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
