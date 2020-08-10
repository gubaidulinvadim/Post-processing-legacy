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
    n_macroparticles = 4*8192
    n_slices=50
    chroma = 0
    i_oct = 550
    machine = FCC(n_segments=1, machine_configuration='50TeV',
                    i_focusing=i_oct, i_defocusing=-i_oct)
    sigma_z = 0.059958
    epsn_x = 2.5e-6  # [m rad]
    epsn_y = 2.5e-6  # [m rad]
    intensity = 1.1e11*5
    bunch = machine.generate_6D_Gaussian_bunch_matched(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

    folder = '/home/vgubaidulin/PhD/Data_server/Stability_scans/FCC/octupoles_test/'
    parameters = {'beta_x': machine.beta_x, 'beta_y': machine.beta_y,
                  'Q_x': machine.Q_x, 'Q_y': machine.Q_y, 'Q_s': machine.Q_s}
    bunch_monitor = get_bunch_monitor(folder, r, i, n_turns)
    slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    slice_monitor = get_slice_monitor(folder, r, i, n_turns, slicer, parameters_dict=parameters)
    particle_monitor = get_particle_monitor(folder, r, i, n_turns, parameters_dict=parameters)
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
    # machine.one_turn_map.append(damperx)
    # machine.one_turn_map.append(dampery)
    sc = TransverseLinearSpaceCharge(slicer, machine.circumference)
    # machine.one_turn_map.append(sc)
    oneslice = slicing.UniformBinSlicer(n_slices=1, n_sigma_z=4)
    L_e = 2
    sigma_e = 4*bunch.sigma_x()
    Ue = 10e3
    gamma_e = 1 + Ue * e / (m_e * c**2)
    beta_e = np.sqrt(1 - gamma_e**-2)
    dQ_max = 0.001
    z = np.linspace(-4*bunch.sigma_z(), 4*bunch.sigma_z(), n_slices)
    I_e = 0.25*np.exp(-z**2/(2*bunch.sigma_z()**2))
    pelens = ElectronLens(L_e, I_e, sigma_e, sigma_e, beta_e, dist='KV')
    # machine.one_turn_map.append(pelens)
    ###########################
    ###########################
    for i in range(n_turns):
        machine.track(bunch)
        bunch_monitor.dump(bunch)
        slice_monitor.dump(bunch)
        # particle_monitor.dump(bunch)


if __name__ == '__main__':
    filename = '/home/vgubaidulin/PhD/Data_server/Stability_scans/FCC/octupoles_test/'
    dQcoh_real = 1e-3*np.linspace(-0.8, 0.8, 17)
    # r=0
    dQcoh_imag = 1e-3*np.linspace(0.0, 0.2, 11)
    np.save(filename+'dQcoh_real', dQcoh_real)
    np.save(filename+'dQcoh_imag', dQcoh_imag)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    points = []
    for r in dQcoh_real:
        for i in dQcoh_imag:
            points.append((r, i))
    Parallel(n_jobs=6)(delayed(run)(r, i) for (r, i) in tqdm(points))
    # run(0., 0.0001)
