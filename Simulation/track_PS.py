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
from PyHEADTAIL.spacecharge.spacecharge import TransverseGaussianSpaceCharge, TransverseLinearSpaceCharge
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.impedances.wakes import WakeField, WakeTable, Resonator, CircularResonator, ParallelPlatesResonator
from PyHEADTAIL.impedances.wakes import ResistiveWall, CircularResistiveWall, ParallelPlatesResistiveWall
from PyHEADTAIL.particles import slicing
from tqdm import tqdm

from helper_funcs import *
from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf

# Simulation parameters declaration


def btf_kick(bunch, phase_vec, Q_x, C):
    kick = 0.0
    Nkick = len(phase_vec)
    dQ = 1e-2
    for j in range(Nkick):
        Q = Q_x-dQ+j*2.0*dQ/Nkick
        kick += (0.05*bunch.sigma_x()/Nkick) * \
            np.cos(2.0*np.pi*Q*c/C+phase_vec[j])
    bunch.xp[:] += kick
    return kick


def run(chromaticity):
    np.random.seed(42)
    n_turns = 2*4*16384
    n_macroparticles = 16384
    machine = PS(n_segments=1, machine_configuration='TOFbeam_transition',
                 Qp_x=chromaticity, Qp_y=chromaticity, gamma=2.49, longitudinal_focusing='non-linear')
    intensity = 3e11
    epsn_x = 2.3e-6
    epsn_y = 2.3e-6
    sigma_z = 180e-9*c/4

    bunch = machine.generate_6D_Gaussian_bunch_matched(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

    folder = '/home/vadim/PhD/Data/CERNPS/'
    bunch_monitor = get_bunch_monitor(folder, chromaticity, n_turns)
    n_slices = 200
    slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    slice_monitor = get_slice_monitor(folder, chromaticity, n_turns, slicer)
    # particle_monitor = get_particle_monitor(folder, r, i, n_turns)
    dt_min = 4*bunch.sigma_z()/c/n_slices

    res_wall1 = CircularResistiveWall(pipe_radius=0.08, resistive_wall_length = 0.57*(531.85+222.3), conductivity=1.4e6, dt_min = dt_min )
    res_wall2 = CircularResistiveWall(pipe_radius=0.07, resistive_wall_length=0.57*332.45, conductivity=1.4e6, dt_min=dt_min)
    wake_field = WakeField(slicer, res_wall1, res_wall2)

    machine.one_turn_map.append(wake_field)
    bunch.x += 0.01*bunch.sigma_x()
    for turn in tqdm(range(n_turns)):
        machine.track(bunch)
        bunch_monitor.dump(bunch)
        slice_monitor.dump(bunch)
        # particle_monitor.dump(bunch)


if __name__ == '__main__':
    filename = '/home/vadim/PhD/Data/CERNPS/'
    chromaticity = np.array((7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2, 1.5, 1.0, 0.75, 0.5,
                             0.25, 0.01, -0.01, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0, -6.0, -7.0))
    # Parallel(n_jobs=-2)(delayed(run)(c) for c in (chromaticity))
    run(0)