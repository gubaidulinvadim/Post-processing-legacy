from __future__ import division

import numpy as np
from scipy.constants import c, e, m_p

from PyCERNmachines.machines import Synchrotron


class FCC(Synchrotron):

    def __init__(self, *args, **kwargs):

        if 'n_segments' not in kwargs.keys():
            raise ValueError('Number of segments must be specified')

        if 'machine_configuration' not in kwargs.keys():
            raise ValueError('machine_configuration must be specified')

        self.n_segments = kwargs['n_segments']
        self.machine_configuration = kwargs['machine_configuration']

        self.circumference  = 100000
        self.s = (np.arange(0, self.n_segments + 1)
                  * self.circumference / self.n_segments)
        
        if self.machine_configuration == '50TeV':
            self.charge = e
            self.mass = m_p

            self.gamma = np.sqrt( (50e12*e/(self.mass*c**2))**2 + 1 )

            self.Q_x     = 111.31
            self.Q_y     = 109.32

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.R/self.Q_x * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.R/self.Q_y * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.Qp_x    = 0
            self.Qp_y    = 0

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha       = 3.225e-4
            self.h1          = 35640
            self.h2          = 71280
            self.V1          = 10e6
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        else:
            raise ValueError('ERROR: unknown machine configuration ' +
                             self.machine_configuration)

        i_focusing = kwargs.pop('i_focusing', False)
        i_defocusing = kwargs.pop('i_defocusing', False)
        if i_focusing or i_defocusing is True:
            #print ('\n--> Powering LHC octupoles to {:g} A.\n'.format(i_focusing))
            self.app_x, self.app_y, self.app_xy = self.get_anharmonicities_from_octupole_currents_FCC(
                i_focusing, i_defocusing)
        super(FCC, self).__init__(*args, **kwargs)
    def get_anharmonicities_from_octupole_currents_FCC(cls, i_focusing, i_defocusing):
        """Calculate the constants of proportionality app_x, app_y and
        app_xy (== app_yx) for the amplitude detuning introduced by the
        LHC octupole magnets (aka. LHC Landau octupoles) from the
        electric currents i_focusing [A] and i_defocusing [A] flowing
        through the magnets. The maximum current is given by
        i_max = +/- 550 [A]. The values app_x, app_y, app_xy obtained
        from the formulae are proportional to the strength of detuning
        for one complete turn around the accelerator, i.e. one-turn
        values.

        The calculation is based on formulae (3.6) taken from 'The LHC
        transverse coupled-bunch instability' by N. Mounet, EPFL PhD
        Thesis, 2012. Values (hard-coded numbers below) are valid for
        LHC Landau octupoles before LS1. Beta functions in x and y are
        correctly taken into account. Note that here, the values of
        app_x, app_y and app_xy are not normalized to the reference
        momentum p0. This is done only during the calculation of the
        detuning in the corresponding detune method of the
        AmplitudeDetuningSegment.

        More detailed explanations and references on how the formulae
        were obtained are given in the PhD thesis (pg. 85ff) cited
        above.
        """
        i_max = 550.  # [A]
        E_max = 50e3 # [GeV]
        L = 0.32
        N_oct = 84*25
        O3 = 63100
        coeff = ( 3*e/(8*np.pi*E_max*1e9*e/c)*N_oct*O3*L )
        beta_x_F =  140/72*175.5
        beta_y_F =  140/72*33.6
        beta_x_D =  140/72*30.1
        beta_y_D =  140/72*178.8
        app_x  = E_max * coeff * (beta_x_F*beta_x_F*i_focusing/i_max - beta_x_D*beta_x_D*i_defocusing/i_max)
        app_xy  = -E_max * coeff * (2*beta_x_F*beta_y_F*i_focusing/i_max - 2*beta_x_D*beta_y_D*i_defocusing/i_max)
        app_y = E_max * coeff * (beta_y_F*beta_y_F*i_focusing/i_max - beta_y_D*beta_y_D*i_defocusing/i_max)

        # Convert to SI units.
        convert_to_SI = e / (1.e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return app_x, app_y, app_xy
