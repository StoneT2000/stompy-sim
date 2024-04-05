from setuptools import find_packages, setup

__version__ = "0.0.1"
setup(
    name="stompy_sim",
    version=__version__,
    description="Simulation of Stompy and various humanoid tasks",
    author="Hillbot",
    url="https://github.com/stonet2000/stompy-sim",
    packages=find_packages(include=["stompy_sim*"]),
    python_requires=">=3.9",
    install_requires=[
        "mani_skill",
        "gymnasium==0.29.1",
        "tyro" # just for cleanrl style ppo code and nice CLIs
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "pre-commit",
            "ipdb",
        ]
    },
)
