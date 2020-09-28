import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from scipy.constants import c, e, m_p, epsilon_0, m_e
from joblib import Parallel, delayed
import sys

import PyHEADTAIL
import PyCERNmachines
from PyCERNmachines.CERNmachines import *
from PyCERNmachines.machines import *
from PyCERNmachines.FCC import *
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.rfq.rfq import RFQTransverseKick, RFQLongitudinalKick
from PyHEADTAIL.multipoles.multipoles import ThinOctupole
from PyHEADTAIL.elens.elens import ElectronLens
from PyHEADTAIL.spacecharge.spacecharge import TransverseGaussianSpaceCharge, TransverseLinearSpaceCharge
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles import slicing
from tqdm import tqdm

from helper_funcs import *
from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf

# Simulation parameters declaration


def run(r, i):
    np.random.seed(42)
    n_turns = 4*8192
    n_macroparticles = 2*16384#8192#16384
    n_slices = 50
    i_oct = 0
    machine = LHC(n_segments=1, machine_configuration='6.5TeV',
                    i_focusing=i_oct, i_defocusing=-i_oct, Qp_x = 0, Qp_y = 0)
    sigma_z = 0.059958
    epsn_x = 2.2e-6  # [m rad]
    epsn_y = 2.2e-6  # [m rad]
    intensity = 1.1e11
    
    bunch = machine.generate_6D_Gaussian_bunch_matched(
       n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
    folder = '/home/vadim/PhD/Data/rfq/'#FCC/octupoles3D/'
    parameters = {'beta_x': machine.beta_x, 'beta_y': machine.beta_y,
                  'Q_x': machine.Q_x, 'Q_y': machine.Q_y, 'Q_s': machine.Q_s}
    bunch_monitor = get_bunch_monitor(folder, r, i, n_turns)
    # slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    # slice_monitor = get_slice_monitor(folder, r, i, n_turns, slicer, parameters_dict=parameters)
    # particle_monitor = get
    # _particle_monitor(folder, r, i, n_turns, parameters_dict=parameters, stride=16)
    phase = 270+np.arctan2(r, i)/(2*np.pi)*360
    tune = np.sqrt(r**2+i**2)
    if tune == 0:
        dampingrate = 0
    else:
        dampingrate = 1/(2*np.pi*tune)
    damperx = TransverseDamper(dampingrate_x=dampingrate, dampingrate_y=None,
                               phase=phase, local_beta_function=machine.beta_x[0], verbose=False)
    dampery = TransverseDamper(dampingrate_x=None, dampingrate_y=dampingrate,
                               phase=phase, local_beta_function=machine.beta_y[0], verbose=False)
    machine.one_turn_map.append(damperx)
    machine.one_turn_map.append(dampery)
    # rfq_long = RFQLongitudinalKick(v_2=4e9, omega=800e6*2.*np.pi, phi_0=0.)
    rfq_trans = RFQTransverseKick(v_2=4e9, omega=800e6*2.*np.pi, phi_0=0.)
    machine.one_turn_map.append(rfq_trans)
    # machine.one_turn_map.append(rfq_long)
    ###########################
    ###########################
    for turn in range(n_turns):
        machine.track(bunch)
        bunch_monitor.dump(bunch)
        # slice_monitor.dump(bunch)
        # particle_monitor.dump(bunch)


if __name__ == '__main__':
    filename = '/home/vadim/PhD/Data/rfq/'
    dQcoh_real = 1e-3*np.linspace(-1.0, 1.0, 21)
    dQcoh_imag = 1e-3*np.linspace(0.0, 0.32, 17)
    np.save(filename+'dQcoh_real', dQcoh_real)
    np.save(filename+'dQcoh_imag', dQcoh_imag)
    points = []
    for r in dQcoh_real:
        for i in dQcoh_imag:
            points.append((r, i))
    Parallel(n_jobs=-2)(delayed(run)(r, i) for (r, i) in tqdm(points))
    # run(0.0000, 0.0000)
    # run(0, 2e-4)