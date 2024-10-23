from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

from monty.dev import requires

from quacc.runners.ase import BaseRunner

has_phonopy = bool(find_spec("phonopy"))
has_seekpath = bool(find_spec("seekpath"))


if TYPE_CHECKING:
    from numpy.typing import NDArray

    if has_phonopy:
        from phonopy import Phonopy


class PhonopyRunner(BaseRunner):
    def __init__(self) -> None:
        """
        Initialize the PhonopyRunner.

        Returns
        -------
        None
        """
        self.setup()

    @requires(has_phonopy, "Phonopy is not installed.")
    @requires(has_seekpath, "Seekpath is not installed")
    def run_phonopy(
        self, phonon: Phonopy, forces: NDArray, symmetrize: bool = False
    ) -> Phonopy:
        """
        Run a phonopy calculation in a temporary directory and
        copy the results to the job results directory.

        Parameters
        ----------
        phonon
            Phonopy object
        forces
            Forces on the atoms
        symmetrize
            Whether to symmetrize the force constants
        Returns
        -------
        Phonopy
            The phonopy object with the results.
        """

        # Run phonopy
        phonon.forces = forces
        phonon.produce_force_constants()

        if symmetrize:
            phonon.symmetrize_force_constants()
            phonon.symmetrize_force_constants_by_space_group()

        phonon.save(
            Path(self.tmpdir, "phonopy.yaml"), settings={"force_constants": True}
        )
        phonon.directory = self.job_results_dir

        # Perform cleanup operations
        self.cleanup()

        return phonon
