from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.mep.neb import NEB as NEB_
from ase.mep.neb import DyNEB as DyNEB_
from ase.mep.neb import NEBOptimizer, idpp_interpolate, interpolate

from quacc import Job, flow, subflow
from quacc.runners.ase import run_neb
from quacc.schemas.ase import summarize_neb_run
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.flow_control import resolve

if TYPE_CHECKING:
    from typing import Any

    from quacc.schemas._aliases.ase import NebSchema


class ConcurrentNEB:
    """
    A class to perform NEB calculations with concurrent force calculations. Using
    the appropriate workflow engine, the forces for all images are calculated
    concurrently (in parallel) without the need for any mpi or threading code.

    As of 04/02/2024 this is experimental and is working with the
    current version of ASE. Tiny changes in the ASE code might break this class.

    In this class the initial and final state are always calculated at the beginning. This is done at the beginning of the calculation. The reason is that it allows for analysis in quacc.runners.ase.summarize_neb

    Since this class actually check for changes BEFORE running the calculation, restarting should be as easy as sending the previous images to this class where
    each image has a calc with results attached.

    **These classes are not meant to be used without any predefined Quacc flow. Unless
    you known what you are doing of course.**
    """

    def __init__(
        self,
        *args,
        force_job,
        force_job_params,
        autorestart_params,
        directory,
        **kwargs,
    ):
        """
        Parameters
        ----------
        force_job
            The force job that will be used to calculate the forces.
        force_job_params
            Parameters to pass to the force job.
        autorestart_params
            When this is not None, autorestart will be turned on and these parameters will be passed to the force_job.
        directory
            For internal use only.
        """

        self.force_job = force_job
        self.force_job_params = force_job_params

        self.autorestart_params = autorestart_params

        self.directory = directory
        self.latest_directories = {}

        super().__init__(*args, **kwargs)

        self.initial_images = self.images.copy()

    def get_forces(self, *args, **kwargs):

        force_jobs, image_idx, to_calc = [], [], []

        for i, image in enumerate(self.images):

            if image.calc is None or image.calc.check_state(image):
                image_idx.append(i)
                to_calc.append(image)

                if i in self.latest_directories:
                    params = recursive_dict_merge(
                        self.force_job_params,
                        self.autorestart_params,
                        {"copy_files": self.latest_directories[i]},
                    )
                else:
                    params = self.force_job_params

                force_jobs.append(partial(self.force_job, **params))

        results = self.concurrent_calculate(to_calc, force_jobs)
        results = resolve(results)

        for idx, result in zip(image_idx, results):
            self.latest_directories[idx] = result["dir_name"]
            self.images[idx].calc = SinglePointDFTCalculator(
                self.images[idx], **result["results"]
            )

        return super().get_forces(*args, **kwargs)

    @subflow
    @staticmethod
    def concurrent_calculate(images, force_jobs):
        return [force_job(atoms=image) for image, force_job in zip(images, force_jobs)]


class NEB(ConcurrentNEB, NEB_):
    pass


class DyNEB(ConcurrentNEB, DyNEB_):
    pass


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

    interpolation_flags = interpolation_params or {}
    autorestart_flags = autorestart_params or {}
    run_flags = run_params or {}
    force_job_flags = force_job_params or {}

    if interpolation_method == "linear":
        interpolate(images, **interpolation_flags)
    elif interpolation_method == "idpp":
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
