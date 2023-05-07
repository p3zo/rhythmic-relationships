from setuptools import find_packages, setup

experimental_requires = ["scikit-learn~=1.0.2", "seaborn~=0.12.0"]

setup(
    name="rhythmic-relationships",
    author="p3zo",
    version="0.1.0",
    url="https://github.com/p3zo/rhythmic-relationships",
    packages=find_packages(
        exclude=[
            "tests*",
        ]
    ),
    install_requires=[
        "bentoml~=1.0.18",
        "matplotlib~=3.5.1",
        "numpy~=1.24.2",
        "pandas~=2.0.0",
        "pillow~=9.5.0",
        "pretty_midi~=0.2.9",
        "pyyaml~=6.0",
        "torch~=1.13.1",
        "torchvision~=0.14.1",
        "tqdm~=4.64.1",
        "wandb~=0.15.1"
    ],
    extras_require={
        "exp": experimental_requires,
    },
)
