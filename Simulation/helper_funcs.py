import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from scipy.constants import c, e, m_p, epsilon_0, m_e
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

import PyHEADTAIL
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.elens.elens import ElectronLens
from PyHEADTAIL.spacecharge.spacecharge import TransverseGaussianSpaceCharge
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.impedances import wakes, wake_kicks
from PyHEADTAIL.aperture.aperture import CircularApertureXY
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles import slicing
from PyHEADTAIL.multipoles.multipoles import ThinOctupole
from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf
# def get_bunch_monitor(folder, tune_real, tune_imag, n_turns, parameters_dict=None):
#     filename = folder+'BM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
#     bunch_monitor = BunchMonitor(
#         filename=filename, n_steps=n_turns, 
#         parameters_dict=parameters_dict, write_buffer_every=1000
#         )
#     return bunch_monitor
# def get_slice_monitor(folder, tune_real, tune_imag, n_turns, slicer, parameters_dict=None):
#     filename = folder+'SLM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
#     slice_monitor = SliceMonitor(filename=filename, n_steps=n_turns, slicer=slicer, parameters_dict=parameters_dict, write_buffer_every=1000)
#     return slice_monitor
# def get_particle_monitor(folder, tune_real, tune_imag, n_turns, parameters_dict=None):
#     filename = folder+'PM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
#     particle_monitor = ParticleMonitor(filename=filename, stide=16, n_steps=n_turns, parameters_dict=parameters_dict, write_buffer_every=1000)
#     return particle_monitor

def get_bunch_monitor(folder, chroma, n_turns, parameters_dict=None):
    filename = folder+'BM(chroma={0:.3f})'.format(chroma)
    bunch_monitor = BunchMonitor(
        filename=filename, n_steps=n_turns, 
        parameters_dict=parameters_dict, write_buffer_every=1000
        )
    return bunch_monitor
def get_slice_monitor(folder, chroma, n_turns, slicer, parameters_dict=None):
    filename = folder+'SLM(chroma={0:.3f})'.format(chroma)
    slice_monitor = SliceMonitor(filename=filename, n_steps=n_turns, slicer=slicer, parameters_dict=parameters_dict, write_buffer_every=1000)
    return slice_monitor
def get_particle_monitor(folder, chroma, n_turns, parameters_dict=None):
    filename = folder+'PM(chroma={0:.3f})'.format(chroma)
    particle_monitor = ParticleMonitor(filename=filename, stide=16, n_steps=n_turns, parameters_dict=parameters_dict, write_buffer_every=1000)
    return particle_monitor

def get_elens(bunch, ratio, dQmax, is_pulsed=False, n_slices=50):
    L_e = 2
    Ue = 10e3
    gamma_e = 1 + Ue * e / (m_e * c**2)
    beta_e = np.sqrt(1 - gamma_e**-2)
    sigma_e = ratio*bunch.sigma_x()
    if is_pulsed:
        z = np.linspace(-4*bunch.sigma_z(), 4*bunch.sigma_z(), n_slices)
        I_e = 1.5*2.5*np.exp(-z**2/(2*bunch.sigma_z()**2))
        elens = ElectronLens(L_e, I_e, sigma_e, sigma_e, beta_e, dist='KV')
    else:
        elens = ElectronLens.RoundDCElectronLens(L_e, dQmax, ratio, beta_e, 'GS', bunch)
    return elens
def get_octupole_kick(N, I, E):
    O3 = 63100*N*I/550
    L = 0.32
    Brho = E/1e9*3.3356
    kL = O3/Brho*L
    return ThinOctupole(kL)
def get_dampers(r, i, beta_x, beta_y):
    phase =  270+np.arctan2(r, i)/(2*np.pi)*360
    tune = np.sqrt(r**2+i**2)
    if tune==0:
        dampingrate=0
    else:
        dampingrate = 1/(2*np.pi*tune)
    damperx = TransverseDamper(dampingrate_x=dampingrate, dampingrate_y=None, phase=phase, local_beta_function=beta_x, verbose=False)
    dampery = TransverseDamper(dampingrate_x = None, dampingrate_y = dampingrate, phase=phase, local_beta_function=beta_y, verbose=False)
    return damperx, dampery