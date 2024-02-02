import numpy as np
from ase.build import minimize_rotation_and_translation
from ase.mep.neb import NEB as NEB_
from ase.mep.neb import NEBState
from ase.optimize.precon import Precon, PreconImages
from quacc import subflow

from quacc.wflow_tools.control import resolve


class ConcurrentNEB:

    def get_forces(self):
        """Evaluate and return the forces."""
        images = self.images

        if not self.allow_shared_calculator:
            calculators = [image.calc for image in images
                           if image.calc is not None]
            if len(set(calculators)) != len(calculators):
                msg = ('One or more NEB images share the same calculator.  '
                       'Each image must have its own calculator.  '
                       'You may wish to use the ase.mep.SingleCalculatorNEB '
                       'class instead, although using separate calculators '
                       'is recommended.')
                raise ValueError(msg)

        forces = np.empty(((self.nimages - 2), self.natoms, 3))
        energies = np.empty(self.nimages)

        if self.remove_rotation_and_translation:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(images[i - 1], images[i])

        if self.method != 'aseneb':
            results = self.concurrent_calculate([images[0], images[-1]], self.force_job)

            results = resolve(results)

            energies[0] = results[0]["results"]["energy"]
            energies[-1] = results[1]["results"]["energy"]
        
        results = self.concurrent_calculate(images[1:-1], self.force_job)

        results = resolve(results)

        for i, result in enumerate(results):
            images[i + 1] = result["atoms"]
            forces[i] = result["results"]["forces"]
            energies[i + 1] = result["results"]["energy"]

        #print(images[1].positions)
        #print(forces)
        #print(energies)
        # if this is the first force call, we need to build the preconditioners
        if self.precon is None or isinstance(self.precon, (str, Precon, list)):
            self.precon = PreconImages(self.precon, images)

        # apply preconditioners to transform forces
        # for the default IdentityPrecon this does not change their values
        precon_forces = self.precon.apply(forces, index=slice(1, -1))

        # Save for later use in iterimages:
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces

        state = NEBState(self, images, energies)

        # Can we get rid of self.energies, self.imax, self.emax etc.?
        self.imax = state.imax
        self.emax = state.emax

        spring1 = state.spring(0)

        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)

            # Get overlap between full PES-derived force and tangent
            tangential_force = np.vdot(forces[i - 1], tangent)

            # from now on we use the preconditioned forces (equal for precon=ID)
            imgforce = precon_forces[i - 1]

            if i == self.imax and self.climb:
                """The climbing image, imax, is not affected by the spring
                   forces. This image feels the full PES-derived force,
                   but the tangential component is inverted:
                   see Eq. 5 in paper II."""
                if self.method == 'aseneb':
                    tangent_mag = np.vdot(tangent, tangent)  # For normalizing
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(state, tangential_force,
                                                tangent, imgforce, spring1,
                                                spring2, i)
                # compute the residual - with ID precon, this is just max force
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)

            spring1 = spring2

        forces = precon_forces.reshape((-1, 3))
    
        if not self.dynamic_relaxation:
            return forces

        """Get NEB forces and scale the convergence criteria to focus
           optimization on saddle point region. The keyword scale_fmax
           determines the rate of convergence scaling."""
        n = self.natoms
        for i in range(self.nimages - 2):
            n1 = n * i
            n2 = n1 + n
            force = np.sqrt((forces[n1:n2] ** 2.).sum(axis=1)).max()
            n_imax = (self.imax - 1) * n  # Image with highest energy.

            positions = self.get_positions()
            pos_imax = positions[n_imax:n_imax + n]

            """Scale convergence criteria based on distance between an
               image and the image with the highest potential energy."""
            rel_pos = np.sqrt(((positions[n1:n2] - pos_imax) ** 2).sum())
            if force < self.fmax * (1 + rel_pos * self.scale_fmax):
                if i == self.imax - 1:
                    # Keep forces at saddle point for the log file.
                    pass
                else:
                    # Set forces to zero before they are sent to optimizer.
                    forces[n1:n2, :] = 0
        return forces
    
    @subflow
    @staticmethod
    def concurrent_calculate(images, force_job):

        return [force_job(atoms=image) for image in images]

class NEB(ConcurrentNEB, NEB_):

    def __init__(self, *args, force_job = None, **kwargs):

        self.force_job = force_job
        super().__init__(*args, **kwargs)
