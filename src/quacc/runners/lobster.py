from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pymatgen.io.lobster import LobsterIn

from quacc.runners.generic import GenericRunner

if TYPE_CHECKING:
    from subprocess import CompletedProcess

    from quacc.types import Filenames, SourceDirectory


class LobsterRunner(GenericRunner):
    """
    A class to run Lobster commands in a subprocess. Inherits from GenericRunner, which handles setup and cleanup of the calculation.
    """

    filepaths: ClassVar[dict[str, SourceDirectory | None]] = {
        "fd_out": None,
        "fd_err": "lobstererr"
    }

    def __init__(
        self,
        command: str,
        copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
        environment: dict[str, str] | None = None,
    ):
        """Initialize the `LobsterRunner` with the command and optional copy files and environment variables."""
        super().__init__(
            command=command, copy_files=copy_files, environment=environment
        )

    def run_lobster(self, lobster_in_dict: dict[str, Any]) -> CompletedProcess:
        """
        Run the Lobster command in a subprocess.

        Parameters
        ----------
        lobster_in_dict
            Dictionary containing the Lobster input parameters.
        """
        lobster_in = LobsterIn(lobster_in_dict)
        lobster_in.write_lobsterin(path=self.tmpdir / "lobsterin")

        return self.run_cmd()
