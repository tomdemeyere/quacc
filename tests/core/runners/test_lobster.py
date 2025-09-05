from __future__ import annotations

from pathlib import Path
from shutil import which
from subprocess import CalledProcessError

import pytest

from quacc import change_settings
from quacc.runners.lobster import LobsterRunner

pytestmark = pytest.mark.skipif(
    which("lobster") is None, reason="lobster not installed"
)

test_files_path = Path(__file__).parent / "test_files"

def test_lobster_runner(tmp_path):
    """Test the `LobsterRunner` class with a simple command."""
    lr = LobsterRunner(command=f"cat {test_files_path / 'lobsterout'}")

    with change_settings({"GZIP_FILES": False}):
        cp = lr.run_lobster(
            {
                "basisSet": "pbeVaspFit 3",
                "cohpGenerator": "COHP",
                "cohpStartEnergy": -15,
                "cohpEndEnergy": 5,
            }
        )

    assert not lr.tmpdir.exists()
    assert lr.job_results_dir.exists()

    assert (lr.job_results_dir / "lobsterin").exists()
    assert cp.stdout.startswith("LOBSTER v5.1.1")

    lines = (lr.job_results_dir / "lobsterin").read_text().splitlines()

    assert lines[0] == "basisSet pbeVaspFit 3"
    assert lines[1] == "cohpGenerator COHP"
    assert lines[2] == "COHPstartEnergy -15"
    assert lines[3] == "COHPendEnergy 5"

