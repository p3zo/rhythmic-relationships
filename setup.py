from setuptools import find_packages, setup

setup(
    name="rhythmic-complements",
    author="p3zo",
    version="0.1.0",
    url="https://github.com/p3zo/rhythmic-complements",
    packages=find_packages(
        exclude=[
            "tests*",
        ]
    ),
    install_requires=[
        "matplotlib~=3.5.1",
        "numpy~=1.23.3",
        "pandas~=1.4.2",
        "PIL~=9.2.0",
        "pretty_midi~=0.2.9",
        "pypianoroll~=1.0.4",
        "torch~=0.13.1",
        "torchvision~=0.14.1",
        "tqdm~=4.64.1",
    ],
)
