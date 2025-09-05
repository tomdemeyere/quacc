"""Summarizer for lobster."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ase.io import read
from pymatgen.io.lobster.inputs import LobsterIn
from pymatgen.io.lobster.outputs import (
    BWDF,
    CHARGE,
    CHARGE_LCFO,
    COBICAR,
    COHPCAR,
    COOPCAR,
    DOSCAR,
    DOSCAR_LCFO,
    GROSSPOP,
    GROSSPOP_LCFO,
    ICOBILIST,
    ICOHPLIST,
    ICOOPLIST,
    POLARIZATION,
    BandOverlaps,
    LobsterMatrices,
    LobsterOut,
    MadelungEnergies,
    NcICOBILIST,
    SitePotential,
)

from quacc import __version__
from quacc.schemas.atoms import atoms_to_metadata
from quacc.utils.dicts import clean_dict
from quacc.utils.files import get_uri

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from pymatgen.io.lobster.core import LobsterFile

    from quacc.types import LobsterResults, LobsterSchema, SourceDirectory

data_sources: dict[str, type[LobsterFile]] = {
    "lobster_out": LobsterOut,
    "charge": CHARGE,
    "charge_lcfo": CHARGE_LCFO,
    "polarization": POLARIZATION,
    "site_potential": SitePotential,
    "band_overlaps": BandOverlaps,
    "bwdf": BWDF,
    "doscar": DOSCAR,
    "doscar_lcfo": DOSCAR_LCFO,
    "grosspop": GROSSPOP,
    "grosspop_lcfo": GROSSPOP_LCFO,
    "madelung_energies": MadelungEnergies,
    "lobster_matrices": LobsterMatrices,
    "cobicar": COBICAR,
    "cohpcar": COHPCAR,
    "coopcar": COOPCAR,
    "icobilist": ICOBILIST,
    "icohplist": ICOHPLIST,
    "icooplist": ICOOPLIST,
    "nc_icobilist": NcICOBILIST,
}

def summarize_lobster(input_atoms: Atoms, directory: SourceDirectory, additional_fields: dict[str, Any] | None = None) -> LobsterSchema:
    """
    Summarize the results of a Lobster calculation.

    Parameters
    ----------
    directory
        The directory containing the Lobster output files.

    Returns
    -------
    LobsterSchema
        A dictionary containing the summarized Lobster results.
    """
    additional_fields = additional_fields or {}

    lobster_in_path = Path(directory, "lobsterin")
    if lobster_in_path.exists():
        lobster_in = LobsterIn.from_file(lobster_in_path)
    else:
        raise FileNotFoundError(f"Lobster input file not found in {directory}")

    results: LobsterResults = cast("LobsterResults", {})

    for name, LobsterClass in data_sources.items():
        default_path = Path(directory, LobsterClass.get_default_filename())

        if default_path.exists():
            results[name] = LobsterClass(filename=default_path)

    inputs = {
        "parameters": dict(lobster_in),
        "nid": get_uri(directory).split(":")[0],
        "dir_name": directory,
        "lobster_metadata": {"version": results["lobster_out"].lobster_version},
        "quacc_version": __version__,
    }

    atoms_metadata = atoms_to_metadata(input_atoms)

    return clean_dict(
        atoms_metadata | inputs | results | additional_fields
    )
