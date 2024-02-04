from __future__ import annotations

from typing import TYPE_CHECKING

from ase.mep.neb import NEBOptimizer, idpp_interpolate, interpolate

from quacc.recipes.common.neb import NEB

from quacc import flow
from quacc.runners.ase import run_neb
from quacc.schemas.ase import summarize_neb_run
from quacc.utils.dicts import recursive_dict_merge

if TYPE_CHECKING:
    from typing import Any

    from quacc import Job
    from quacc.schemas._aliases.ase import NebSchema


@flow
def neb_flow(
    images,
    force_job: Job,
    force_job_params: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    run_params: dict[str, Any] | None = None,
    autorestart_params: dict[str, Any] | None = None,
    interpolation_method: str = "idpp",
    interpolation_params: dict[str, Any] | None = None,
    **neb_kwargs,
) -> NebSchema:
    """
    Function to perform a NEB calculation. Using the Quacc monkeypatched NEB.

    Please see the NEB class above for more information. Small important details:

    - The NEB implementation is change by setting the neb_class parameter to one
    of the two classes above. AutoNEB is not implemented yet.

    - autorestart_params is a dictionary, these parameters will be merged to force_job_params when the NEB calculation have done more than one iteration.
    This can be useful to turn on any DFT restart functionality in the force_job.
    Similarly, when this keyword is not None, the directories of the previous calculations will be copied to the new force job, image by image.

    - interpolation_method is a string, either "linear" or "idpp". If none, images are taken as is.

    - Some NEB optimizers (NEBOptimizer) do not allow for an easy check of convergence,
    if you have activated CHECK_CONVERGENCE in the Quacc settings, this check will be
    skipped for these optimizers and the NEB will always be considered converged.


    Parameters
    ----------
    images
        List of images.
    force_job
        The force job that will be used to calculate the forces.
    force_job_params
        Parameters to pass to the force job.
    opt_params
        Parameters to pass to the optimizer.
    run_params
        Parameters to pass to the run call.
    autorestart_params
        When this is not None, autorestart will be turned on and these parameters will be passed to the force_job.
    interpolation_method
        The interpolation method to use. Either "linear" or "idpp". If none,
        images are taken as is.
    interpolation_kwargs
        Additional kwargs to pass to the interpolation method.
    neb_kwargs
        Additional kwargs to pass to the NEB class.

    Returns
    -------
    NebSchema
        The results of the NEB calculation.
    """

    autorestart_flags = autorestart_params or {}
    run_flags = run_params or {}
    force_job_flags = force_job_params or {}

    interpolation_flags = interpolation_params or {}

    if interpolation_method == "linear":
        interpolate(images, **interpolation_flags)
    elif interpolation_method == "idpp":
        interpolation_defaults = {
            "fmax": 0.005,
            "steps": 9999,
            "traj": None,
            "log": None,
        }
        interpolation_flags = recursive_dict_merge(
            interpolation_defaults, interpolation_flags
        )
        idpp_interpolate(images, **interpolation_flags)

    neb_defaults = {"neb_class": NEB, "method": "aseneb", "climb": False}
    neb_flags = recursive_dict_merge(neb_defaults, neb_kwargs)

    opt_defaults = {"optimizer": NEBOptimizer}
    optimizer_flags = recursive_dict_merge(opt_defaults, opt_params)

    run_defaults = {"fmax": 0.05, "steps": 1000}
    run_flags = recursive_dict_merge(run_defaults, run_params)

    neb, dyn = run_neb(
        images,
        force_job,
        force_job_flags,
        neb_flags,
        optimizer_flags,
        run_flags,
        autorestart_flags,
    )

    return summarize_neb_run(neb, dyn)
