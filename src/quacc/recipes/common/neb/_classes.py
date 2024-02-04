from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.mep.neb import NEB as NEB_
from ase.mep.neb import DyNEB as DyNEB_

from quacc import subflow
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.flow_control import resolve

if TYPE_CHECKING:

    pass


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
