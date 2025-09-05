"""Common workflows for phonons."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quacc import Job, get_settings, job, subflow
from quacc.runners.lobster import LobsterRunner
from quacc.schemas.lobster import summarize_lobster
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.customizers import customize_funcs

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase.atoms import Atoms

    from quacc import Job
    from quacc.types import Filenames, LobsterSchema, SourceDirectory


@job
def lobster_job(
    atoms: Atoms,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    lobster_in_dict: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> LobsterSchema:
    additional_fields = additional_fields or {}
    lobster_in_dict = lobster_in_dict or {}

    command = get_settings().LOBSTER_CMD or "lobster"
    runner = LobsterRunner(command, copy_files=copy_files)

    runner.run_lobster(lobster_in_dict)

    results = summarize_lobster(
        input_atoms=atoms,
        directory=runner.tmpdir,
        additional_fields=additional_fields,
    )

    runner.cleanup()

    return results


@subflow
def lobster_subflow(
    atoms: Atoms,
    nscf_job: Job | None = None,
    lobster_in_dict: dict[str, str] | None = None,
    additional_fields: dict[str, Any] | None = None,
    job_params: dict[str, Any] | None = None,
    job_decorators: dict[str, Callable | None] | None = None,
) -> LobsterSchema:
    job_params = job_params or {}

    default_job_params = {
        "nscf_job": recursive_dict_merge({
            "input_data": {
                "control": {"wf_collect": True}, "system": {"nosym": True}
            }
        }, job_params.get("nscf_job")),
    }

    nscf_job = customize_funcs(
        ["nscf_job"],
        [nscf_job],
        param_defaults=default_job_params,
        param_swaps=job_params,
        decorators=job_decorators,
    )

    results = nscf_job(atoms, copy_files=results["dir_name"])

    return lobster_job(
        atoms,
        results["dir_name"],
        lobster_in_dict=lobster_in_dict,
        additional_fields=additional_fields,
    )
