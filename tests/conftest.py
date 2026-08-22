import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(REPO_ROOT, "input", "babyslakh")


@pytest.fixture(scope="session")
def prepared_dataset(tmp_path_factory):
    """Slice the BabySlakh input the repo ships into a dataset the Dataset classes can load.

    Takes a few seconds, so it is built once per session. Prepared into a tmpdir rather than
    `output/` so the tests do not depend on what a previous run happened to leave behind.
    """
    datasets_dir = tmp_path_factory.mktemp("datasets")

    subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "scripts", "prepare_dataset.py"),
            f"--path={INPUT_DIR}",
            f"--outdir={datasets_dir}",
            "--prefix=babyslakh",
            "--seg_size=1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    n_files = len([f for f in os.listdir(INPUT_DIR) if f.endswith(".mid")])
    dataset_name = f"babyslakh_{n_files}_1bar_4res"

    return str(datasets_dir), dataset_name
