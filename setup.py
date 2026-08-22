from setuptools import find_packages, setup

# `rhythmtoolbox` is not published to PyPI, so it is referenced directly. This makes the
# distribution uninstallable from a package index, which is fine for a research repo.
RHYTHMTOOLBOX = "rhythmtoolbox @ git+https://github.com/danielgomezmarin/rhythmtoolbox"

# `rhythmtoolbox` pins numpy~=1.24.2 and scipy~=1.10.1, which caps everything downstream of them
install_requires = [
    "matplotlib~=3.7.0",
    "numpy~=1.24.2",
    "pandas~=2.0.0",
    "pillow>=9.5.0",
    "pretty_midi~=0.2.10",
    "pyyaml~=6.0",
    RHYTHMTOOLBOX,
    "scipy~=1.10.1",
    "seaborn~=0.12.0",
    "torch>=2.0",
    "tqdm~=4.64",
    "wandb>=0.15",
    "x-transformers>=1.16,<3",
]

# Only needed by the analysis scripts, not by the library
experimental_requires = [
    "scikit-learn~=1.3.0",
]

# Only needed by `model_utils.save_bento_model`, which imports it lazily
bento_requires = [
    "bentoml~=1.0.18",
]

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
    install_requires=install_requires,
    extras_require={
        "experimental": experimental_requires,
        "bento": bento_requires,
        "test": ["pytest", "pytest-cov"],
    },
)
