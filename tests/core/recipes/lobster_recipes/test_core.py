from __future__ import annotations

from shutil import which

import pytest

from quacc import get_settings

pytestmark = pytest.mark.skipif(
    which(str(get_settings().ESPRESSO_BINARIES["pw"])) is None,
    reason="QE not installed",
)

from pathlib import Path
from shutil import which

import pytest
from ase.build import bulk
from ase.optimize import BFGS
from monty.io import zopen
from numpy.testing import assert_allclose, assert_array_equal

from quacc import JobFailure
from quacc.calculators.espresso.espresso import EspressoTemplate
from quacc.recipes.espresso.core import (
    ase_relax_job,
    non_scf_job,
    post_processing_job,
    relax_job,
    static_job,
)
from quacc.recipes.common.lobster import lobster_subflow

from functools import partial
from quacc.utils.files import copy_decompress_files

DATA_DIR = Path(__file__).parents[1] / "espresso_recipes" / "data"

def test_lobster_subflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    copy_decompress_files(DATA_DIR, ["Si.upf.gz"], tmp_path)

    atoms = bulk("Si")

    pseudopotentials = {"Si": "Si.upf"}
    input_data = {"control": {"pseudo_dir": tmp_path}}

    static_job_custom = partial(static_job, input_data=input_data, pseudopotentials=pseudopotentials)
    nscf_job_custom = partial(non_scf_job, input_data=input_data, pseudopotentials=pseudopotentials)

    lobster_subflow(
        atoms,
        scf_job=static_job_custom,
        nscf_job=nscf_job_custom,
        lobster_in_dict={
            "basisSet": "pbeVaspFit 3",
            "cohpGenerator": "COHP",
            "cohpStartEnergy": -15,
            "cohpEndEnergy": 5,
        },
    )

