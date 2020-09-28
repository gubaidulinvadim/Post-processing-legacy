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
    n_turns = 8*16384
    n_turns_slicemonitor = 4096
    n_macroparticles = 32*16384
    intensity = 5e12
    n_segments = 1
    Ekin = 4.5e9
    gamma = 1 + Ekin * e / (m_p * c**2)
    C = 261.72
    A = 1
    Z = 1
    Q_x = 4.2
    Q_y = 3.4
    # Q_s = 0.0002
    # eta = 0.06
    alpha_0 = [3.0]
    # alpha_0.append((eta+1/gamma**2))
    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj = C/(2*np.pi*Q_x)
    beta_y_inj = C/(2*np.pi*Q_y)
    sigma_z = 4
    epsn_x = 12.5e-6  # [m rad]
    epsn_y = 12.5e-6  # [m rad]
    long_map = RFSystems(C, [1, ], [16e3, ], [0, ],
                         alpha_0, gamma, mass=A*m_p, charge=Z*e)

    beta = np.sqrt(1 - gamma**-2)
    R = C / (2.*np.pi)
    gamma_t = 1. / np.sqrt(alpha_0)
    eta = alpha_0[0] - 1. / gamma**2

    p0 = np.sqrt(gamma**2 - 1) * A * m_p * c
    beta_z = (long_map.eta(dp=0, gamma=gamma) *
              long_map.circumference / (2 * np.pi * long_map.Q_s))
    epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)
    # Parameters for transverse map.
    s = np.arange(0, n_segments + 1) * C / n_segments
    alpha_x = alpha_x_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)
    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_y = np.zeros(n_segments)

    bunch = generators.generate_Gaussian6DTwiss(
        macroparticlenumber=n_macroparticles, intensity=intensity, charge=Z*e,
        gamma=gamma, mass=A*m_p, circumference=C,
        alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
        alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z,  limit_n_rms_x=3.5, limit_n_rms_y=3.5)

    folder = '/home/vgubaid/SIS18/'#'/home/vadim/PhD/Data/SIS18/'
    bunch_monitor = get_bunch_monitor(folder, chromaticity, n_turns)
    n_slices = 250
    slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    slice_monitor = get_slice_monitor(folder, chromaticity, n_turns_slicemonitor, slicer)
    # particle_monitor = get_particle_monitor(folder, r, i, n_turns)
    dt_min = 4*bunch.sigma_z()/c/n_slices
    res_wall1 = CircularResistiveWall(pipe_radius=75e-3,
                                      resistive_wall_length=3*30.72,
                                      dt_min=dt_min,
                                      conductivity=1.4e6)
    res_wall2 = CircularResistiveWall(pipe_radius=75e-3,
                                      resistive_wall_length=3*62.88,
                                      dt_min=dt_min,
                                      conductivity=1.4e6)
    res_wall3 = CircularResistiveWall(pipe_radius=0.1,
                                      resistive_wall_length=3*123.12,
                                      dt_min=dt_min,
                                      conductivity=1.4e6)
    wake_field = WakeField(slicer, res_wall1)
    wake_field2 = WakeField(slicer, res_wall2)
    wake_field3 = WakeField(slicer, res_wall3)
    # rfq_long = RFQLongitudinalKick(v_2=1e5, omega=2*np.pi*5.6e6, phi_0=0)
    # rfq = RFQTransverseKick(v_2=1e5, omega=2*np.pi*5.6e6, phi_0=0)
    chroma = Chromaticity(Qp_x=chromaticity, Qp_y=chromaticity)
    trans_map = TransverseMap(s, alpha_x, beta_x, D_x,
                              alpha_y, beta_y, D_y, Q_x, Q_y, [chroma])
    trans_one_turn = [m for m in trans_map]
    map_ = trans_one_turn + [long_map, wake_field, wake_field2, wake_field3]
    # kicks = np.empty((n_turns,), dtype=np.float64)
    # phase_vec = 2*np.pi*np.random.random_sample(10000)
    # amplitude = 1e-2*bunch.sigma_x()
    bunch.x += 0.01*bunch.sigma_x()
    for turn in tqdm(range(n_turns)):
        for m_ in map_:
            m_.track(bunch)
        # kicks[turn] = btf_kick(bunch, phase_vec, 4.2, 261.72)
        bunch_monitor.dump(bunch)
        if (turn >= n_turns - n_turns_slicemonitor):
            slice_monitor.dump(bunch)
        # particle_monitor.dump(bunch)
    # np.save('/home/vgubaidulin/PhD/Data_server/Stability_scans/SIS18_instability/kicks.npy', kicks)


if __name__ == '__main__':
    # filename = #'/home/vadim/PhD/Data/SIS18/'
    chromaticity = np.array((7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2, 1.5, 1.0, 0.75, 0.5,
                             0.25, 0.01, -0.01, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0, -6.0, -7.0))
    # chromaticity = np.array((0.01, 0.5, 1.0, 2.0, 4.0, 6.0))
    Parallel(n_jobs=-2)(delayed(run)(c) for c in (chromaticity))