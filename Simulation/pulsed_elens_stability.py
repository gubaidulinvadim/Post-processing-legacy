from __future__ import division, print_function
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
from PyHEADTAIL.elens.elens import ElectronLens
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor

from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles import slicing

from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf

#Simulation parameters declaration
n_macroparticles = 4*16384

def get_bunch_monitor(folder, tune_real, tune_imag, n_turns, Q_x, Q_y):
    filename_appdx='(dQreal={0:.3f},dQimag={1:.3f}'.format(tune_real*1e3, tune_imag*1e3)
    bunch_filename = folder+'bunch_monitor(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    parameters_dict = {'Q_x': Q_x, 'Q_y': Q_y, 'turns': n_turns, 'macroparticles': n_macroparticles}
    bunch_monitor = BunchMonitor(
        filename=bunch_filename, n_steps=n_turns, 
        parameters_dict=parameters_dict, write_buffer_every=1000
        )
    return bunch_monitor
def get_particle_monitor(folder, tune_real, tune_imag, n_turns, Q_x, Q_y):
    filename_appdx='(dQreal={0:.3f},dQimag={1:.3f}'.format(tune_real*1e3, tune_imag*1e3)
    filename = folder+'particle_monitor(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    parameters_dict = {'Q_x': Q_x, 'Q_y': Q_y, 'turns': n_turns, 'macroparticles': n_macroparticles}
    particle_monitor = ParticleMonitor(
        filename=filename, stride=10, parameters_dict=parameters_dict
    )
    return particle_monitor
def initialize_damper(dQre, dQim, beta_x, beta_y):
    phase = 270+180/np.pi*np.arctan2(dQre, dQim)
    tune = np.sqrt(dQre*dQre+dQim*dQim) 
    damping_rate = 1e3/(2*np.pi*tune) if tune != 0 else 0
    print('Damping rate is: ', damping_rate)
    damperx = TransverseDamper(dampingrate_x=damping_rate,
     dampingrate_y=None,
      phase=phase,
       local_beta_function=beta_x,
        verbose=False)
    dampery = TransverseDamper(dampingrate_x=None,
     dampingrate_y=damping_rate,
      phase=phase,
       local_beta_function=beta_y,
        verbose=False) 
    return damperx, dampery
def initialize_elens(slicer, bunch):
    L_e = 2
    n_slices = 50
    z = np.linspace(-4*bunch.sigma_z(), 4*bunch.sigma_z(), n_slices)
    I_e = 0.25*np.exp(-z**2/(2*bunch.sigma_z()**2))
    sigma_e = 1.0*bunch.sigma_x()
    Ue = 10e3
    gamma_e = 1 + Ue * e / (mcd Code/Stability_scans/_e * c**2)
    beta_e = np.sqrt(1 - bunch.gamma**-2)
    pelens = ElectronLens(L_e, I_e, sigma_e, sigma_e, beta_e, dist='KV')
    return pelens

def run(r, i):
    n_turns = 16384#*8
    np.random.seed(42)

    #Machine parameters declaration
    n_segments = 1
    Ekin = 50e12
    intensity = 1e11
    C = 100000

    Q_x = 111.31
    Q_y = 109.32
    Q_s = 0.0012

    alpha_0 = [0.0003225]
    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj =  92.7 * (141 / 72)
    beta_y_inj = 93.2 * (141 / 72)

    sigma_z = 0.059958
    epsn_x = 2.2e-6 
    epsn_y = 2.2e-6 
    
    long_map = LinearMap(alpha_0, C, Q_s)
    gamma = 1 + Ekin * e / (m_p * c**2)
    beta = np.sqrt(1 - gamma**-2)
    R = C / (2.*np.pi)
    gamma_t = 1. / np.sqrt(alpha_0)
    eta = alpha_0[0] - 1. / gamma**2
    p0 = np.sqrt(gamma**2 - 1) * m_p * c
    beta_z = (long_map.eta(dp=0, gamma=gamma) * long_map.circumference / (2 * np.pi * long_map.Q_s))
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
    macroparticlenumber=n_macroparticles,
    intensity=intensity,
    charge=e,
    mass=m_p,
    circumference=C,
    gamma=gamma,
    alpha_x = alpha_x, alpha_y = alpha_y, beta_x=beta_x, beta_y=beta_y, beta_z=beta_z, epsn_x = epsn_x, epsn_y = epsn_y, epsn_z = epsn_z, 
    limit_n_rms_x=3.5, limit_n_rms_y=3.5)
    damper_x, damper_y = initialize_damper(r, i, beta_x_inj, beta_y_inj)
    trans_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    trans_one_turn = [m for m in trans_map]
    
    n_slices=50
    slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    
    pelens = initialize_elens(slicer, bunch)
    map_ = trans_one_turn  + [long_map, damper_x, damper_y, pelens]
    folder = '/home/vgubaidulin/PhD/Data/Stability_scans/pulsed_elens/'
    bunch_monitor = get_bunch_monitor(folder, r, i, n_turns, Q_x, Q_y)
    particle_monitor = get_particle_monitor(folder, r, i, n_turns, Q_x, Q_y)
    for turn in range(n_turns): 
        for m in map_:
            m.track(bunch)
        # bunch_monitor.dump(bunch)    
        # particle_monitor.dump(bunch)
if __name__=='__main__':
    
    filename = '/home/vgubaidulin/PhD/Data/Stability_scans/pulsed_elens/'
    dQre = 1e-3*np.linspace(-0.6, 0.6, 25)
    dQim = 1e-3*np.linspace(0.0, 2.5, 26)
    np.save(filename+'dQre', dQre)
    np.save(filename+'dQim', dQim)
    points = []
    for r in dQre:
        for i in dQim:
            points.append((r, i))
    # Parallel(n_jobs=-1)(delayed(run)(r, i) for (r, i) in tqdm(points)) 
