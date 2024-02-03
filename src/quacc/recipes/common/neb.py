from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep.neb import NEB as NEB_
from ase.mep.neb import DyNEB as DyNEB_
from ase.mep.neb import NEBOptimizer, idpp_interpolate, interpolate

from quacc import Job, flow, subflow
from quacc.runners.ase import run_neb
from quacc.schemas.ase import summarize_neb
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.flow_control import resolve

if TYPE_CHECKING:
    from typing import Any

    from quacc.schemas._aliases.ase import NebSchema


class ConcurrentNEB:

    def __init__(
        self,
        *args,
        force_job=None,
        force_job_params,
        autorestart_params,
        directory,
        **kwargs,
    ):

        self.force_job = force_job
        self.force_job_params = force_job_params

        self.autorestart_params = autorestart_params

        self.latest_directories = {}

        self.directory = directory

        super().__init__(*args, **kwargs)

    def get_forces(self, *args, **kwargs):

        sl = slice(0, len(self.images)) if self.method != "aseneb" else slice(1, -1)

        to_calc = []

        force_jobs = []

        image_idx = []

        for i, image in zip(list(range(len(self.images)))[sl], self.images[sl]):

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

            self.images[idx].calc = SinglePointCalculator(
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
    autorestart_params
        When this is not None, autorestart will be turned on and these parameters will be passed to the force_job.
    interpolation_method
        The interpolation method to use. Either "linear" or "idpp". If none,
        images are taken as is.
    interpolation_kwargs
        Additional kwargs to pass to the interpolation method.

    Returns
    -------
    RunSchema
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

    neb_defaults = {"class": NEB, "method": "aseneb", "climb": False}
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

    return summarize_neb(neb, dyn)
