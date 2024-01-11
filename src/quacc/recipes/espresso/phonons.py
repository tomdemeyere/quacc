"""
This module, 'phonons.py', contains recipes for performing phonon calculations
using the ph.x binary from Quantum ESPRESSO via the quacc library. The recipes
provided in this module are jobs and flows that can be used to perform phonon
calculations in different fashion.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io.espresso import Namelist

from quacc import Job, flow, job, subflow
from quacc.calculators.espresso.espresso import EspressoTemplate
from quacc.recipes.espresso._base import base_fn
from quacc.recipes.espresso.core import relax_job
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.customizers import customize_funcs, strip_decorator

if TYPE_CHECKING:
    from typing import Any, Callable

    from ase.atoms import Atoms

    from quacc.schemas._aliases.ase import RunSchema


@job
def phonon_job(
    prev_dir: str | Path,
    parallel_info: dict[str] | None = None,
    test_run: bool = False,
    **calc_kwargs,
) -> RunSchema:
    """
    Function to carry out a basic ph.x calculation. It should allow you to
    use all the features of the [ph.x binary](https://www.quantum-espresso.org/Doc/INPUT_PH.html)

    This job requires the results of a previous pw.x calculation, you might
    want to create your own flow to run both jobs in sequence.

    Parameters
    ----------
    prev_dir
        Outdir of the previously ran pw.x calculation. This is used to copy
        the entire tree structure of that directory to the working directory
        of this calculation.
    parallel_info
        Dictionary containing information about the parallelization of the
        calculation. See the ASE documentation for more information.
    test_run
        If True, a test run is performed to check that the calculation input_data is correct or
        to generate some files/info if needed.
    **calc_kwargs
        calc_kwargs dictionary possibly containing the following keys:

        - input_data: dict
        - qpts: list[list[float]] | list[tuple[float]] | list[float]
        - nat_todo: list[int]

        See the docstring of `ase.io.espresso.write_espresso_ph` for more information.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.summarize_run][]
    """

    calc_defaults = {
        "input_data": {
            "inputph": {"tr2_ph": 1e-12, "alpha_mix(1)": 0.1, "verbosity": "high"}
        },
        "qpts": (0, 0, 0),
    }

    return base_fn(
        template=EspressoTemplate("ph", test_run=test_run),
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        parallel_info=parallel_info,
        additional_fields={"name": "ph.x Phonon"},
        copy_files=prev_dir,
    )


@flow
def grid_phonon_flow(
    atoms: Atoms,
    nblocks: int = 1,
    job_decorators: dict[str, Callable | None] | None = None,
    job_params: dict[str, Any] | None = None,
) -> RunSchema:
    """
    This function performs grid parallelization of a ph.x calculation. The grid
    parallelization is a technique to make phonon calculation embarrassingly
    parallel. Each representation of each q-point is calculated in a separate job,
    allowing for distributed computation across different machines and times.
    This function should return similar results to [quacc.recipes.espresso.phonons.phonon_job][]
    for similiar system and settings. If you don't know about grid parallelization please consult
    the Quantum Espresso user manual and exemples.

    Instructions for the impatient
    ------------------------------
    If you never used the Espresso interface for quacc please visit []
    
    Unless you have specific needs, this flow only requires to change the parameters
    of the "ph_job" in job_params. For more information about job_decorators and job_params
    please visit []

    One simple example:

    ``` python
    from quacc.recipes.espresso.phonons import grid_phonon_flow

    grid_flow_params = {"ph_job": {"input_data": {"ldisp": True, "nq1": 8, "nq2": 8, "nq3": 8}}}
    grid_phonon_flow(atoms=ase.build.bulk('Pt'), job_params=grid_flow_params)
    ```
    **Do not change the "lqdir", "recover" and "low_directory_check" keywords if you don't
    have a very good reason.**

    Disk space management
    ---------------------
    By default this flow is set to use maximum disk space for performance trade-off.
    Calculated properties during the initialization phase will be copied to each
    representation. This is approximatively twice the space taken by pw.x
    for each representation.

    To mitigate this, two options:

    - The "nblocks" parametrer can be provided. Multiple representations will be then
    grouped together in single job, this reduces parallelization level, but will use
    less space accordingly.

    - A much less desirable options is to modify the _grid_phonon_subflow to avoid copying
    the ph.x calculated band-structures. This will be horribly inefficient since the electric-part
    + the band structure will have to be recomputed for each job, but will halve disk space usage.

    Example: pw.x: 20 GB, 120 representations will lead to: 20*2*120 = 4.8 TB. Using nblocks = 3
    will lead to a division by a factor of 3: 20*2*120/3 = 1.6TB

    Jobs
    ----
    1. pw.x relaxation
        - name: "relax_job"
        - job: [quacc.recipes.espresso.core.relax_job][]

    2. ph.x initialization phase.
        - name: "ph_init_job"
        - job: [quacc.recipes.espresso.phonons.phonon_job][]

    3. Core ph.x calculations
        - name: "ph_job"
        - job: [quacc.recipes.espresso.phonons.phonon_job][]

    4. ph.x calculation to gather data and diagonalize each dynamical matrix
        - name: "ph_recover_job"
        - job: [quacc.recipes.espresso.phonons.phonon_job][]

    Parameters
    ----------
    atoms
        Atoms object
    nblocks
        The number of representations to group together in a single job.
        This will reduce the amount of data produced by a factor of nblocks.
        If nblocks = 0, each job will contain all the representations for a
        single q-point.
    job_params
        Custom parameters to pass to each Job in the Flow. This is a dictionary where
        the keys are the names of the jobs and the values are dictionaries of parameters.
    job_decorators
        Custom decorators to apply to each Job in the Flow. This is a dictionary where
        the keys are the names of the jobs and the values are decorators.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.summarize_run][]
    """

    @job
    def _ph_recover_job(grid_results: list[RunSchema]) -> RunSchema:
        prev_dirs = {}
        for result in grid_results:
            prev_dirs[result["dir_name"]] = [
                "**/*.xml.*",
                "**/data-file-schema.xml.*",
                "**/charge-density.*",
                "**/wfc*.*",
                "**/paw.txt.*",
            ]
        return strip_decorator(ph_recover_job)(prev_dirs)

    @subflow
    def _grid_phonon_subflow(
        ph_input_data: dict | None,
        ph_init_job_results: str | Path,
        ph_job: Job,
        nblocks: int = 1,
    ) -> list[RunSchema]:
        """
        This functions is a subflow used in [quacc.recipes.espresso.phonons.grid_phonon_flow][].

        Parameters
        ----------
        ph_input_data
            The input data for the phonon calculation.
        ph_init_job_results
            The results of the phonon 'only_init' job.
        ph_job
            The phonon job to be executed.
        nblocks
            The number of blocks for grouping representations. Defaults to 1.

        Returns
        -------
        list[RunSchema]
            A list of results from each phonon job.
        """

        ph_input_data = Namelist(ph_input_data)
        ph_input_data.to_nested(binary="ph")

        prefix = ph_input_data["inputph"].get("prefix", "pwscf")
        outdir = ph_input_data["inputph"].get("outdir", ".")
        lqdir = ph_input_data["inputph"].get("lqdir", False)

        grid_results = []
        for n, (qpoint, qdata) in enumerate(ph_init_job_results["results"].items()):
            ph_input_data["inputph"]["start_q"] = n + 1
            ph_input_data["inputph"]["last_q"] = n + 1
            this_block = nblocks if nblocks > 0 else len(qdata["representations"])
            repr_to_do = [
                r
                for r in qdata["representations"]
                if not qdata["representations"][r]["done"]
            ]
            repr_to_do = np.array_split(
                repr_to_do, np.ceil(len(repr_to_do) / this_block)
            )
            file_to_copy = {
                ph_init_job_results["dir_name"]: [
                    f"{outdir}/{prefix}.save/charge-density.*",
                    f"{outdir}/{prefix}.save/data-file-schema.xml.*",
                    f"{outdir}/{prefix}.save/paw.txt.*",
                    f"{outdir}/{prefix}.save/wfc*.*",
                ]
            }
            if qpoint != (0.0, 0.0, 0.0) and lqdir:
                file_to_copy[ph_init_job_results["dir_name"]].extend(
                    [
                        f"{outdir}/_ph0/{prefix}.q_{n + 1}/{prefix}.save/*",
                        f"{outdir}/_ph0/{prefix}.q_{n + 1}/{prefix}.wfc*",
                        f"{outdir}/_ph0/{prefix}.phsave/control_ph.xml*",
                        f"{outdir}/_ph0/{prefix}.phsave/status_run.xml*",
                        f"{outdir}/_ph0/{prefix}.phsave/patterns.*.xml*",
                    ]
                )
            for representation in repr_to_do:
                ph_input_data["inputph"]["start_irr"] = representation[0]
                ph_input_data["inputph"]["last_irr"] = representation[-1]
                ph_job_results = ph_job(file_to_copy, input_data=ph_input_data)
                grid_results.append(ph_job_results)

        return grid_results

    calc_defaults = {
        "relax_job": {
            "input_data": {
                "control": {"forc_conv_thr": 5.0e-5},
                "electrons": {"conv_thr": 1e-12},
            }
        },
        "ph_init_job": recursive_dict_merge(
            {"input_data": {"inputph": {"lqdir": True, "only_init": True}}},
            job_params["ph_job"],
        ),
        "ph_job": {
            "input_data": {
                "inputph": {"lqdir": True, "low_directory_check": True, "recover": True}
            }
        },
        "ph_recover_job": recursive_dict_merge(
            {"input_data": {"inputph": {"recover": True, "lqdir": True}}},
            job_params["ph_job"],
        ),
    }

    job_params = recursive_dict_merge(calc_defaults, job_params)

    pw_job, ph_init_job, ph_job, ph_recover_job = customize_funcs(
        ["relax_job", "ph_init_job", "ph_job", "ph_recover_job"],
        [relax_job, phonon_job, phonon_job, phonon_job],
        parameters=job_params,
        decorators=job_decorators,
    )

    pw_job_results = pw_job(atoms)

    ph_init_job_results = ph_init_job(pw_job_results["dir_name"])

    grid_results = _grid_phonon_subflow(
        job_params["ph_job"]["input_data"], ph_init_job_results, ph_job, nblocks=nblocks
    )

    return _ph_recover_job(grid_results)
