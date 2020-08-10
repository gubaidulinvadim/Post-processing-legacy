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
from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf
def get_bunch_monitor(folder, tune_real, tune_imag, n_turns, parameters_dict=None):
    filename = folder+'BM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    bunch_monitor = BunchMonitor(
        filename=filename, n_steps=n_turns, 
        parameters_dict=parameters_dict, write_buffer_every=20
        )
    return bunch_monitor
def get_slice_monitor(folder, tune_real, tune_imag, n_turns, slicer, parameters_dict=None):
    filename = folder+'SLM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    slice_monitor = SliceMonitor(filename=filename, n_steps=n_turns, slicer=slicer, parameters_dict=parameters_dict, write_buffer_every=20)
    return slice_monitor
def get_particle_monitor(folder, tune_real, tune_imag, n_turns, parameters_dict=None):
    filename = folder+'PM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    particle_monitor = ParticleMonitor(filename=filename, stide=10, n_steps=n_turns, parameters_dict=parameters_dict, write_buffer_every=20)
    return particle_monitor
    