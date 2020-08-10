from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from scipy.constants import c, e, m_p, epsilon_0, m_e
from joblib import Parallel, delayed
import sys

import PyHEADTAIL
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.spacecharge.spacecharge import TransverseGaussianSpaceCharge
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.impedances import wakes, wake_kicks
from PyHEADTAIL.aperture.aperture import CircularApertureXY
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles import slicing

from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf

#Simulation parameters declaration
n_turns = 8192
n_macroparticles = 4*16384
def get_real_coherent_tune_shift(x_rec, y_rec):
    coh_x = pnf.naff(x_rec, turns=n_turns)[0, 1]
    coh_y = pnf.naff(y_rec, turns=n_turns)[0, 1] 
    return coh_x, coh_y
def get_imaginary_coherent_tune_shift(x_rec, y_rec, px_rec, py_rec, beta_x, beta_y):
    Jx = np.sqrt(x_rec*x_rec+beta_x*px_rec*px_rec)
    Jy = np.sqrt(y_rec*y_rec+beta_y*py_rec*py_rec)
    t = np.arange(n_turns)
    coh_x, amplitude_x = np.polyfit(t, np.log(2 * Jx), 1)
    coh_y, amplitude_y = np.polyfit(t, np.log(2 * Jy), 1)
    return coh_x/(2*np.pi), coh_y/(2*np.pi)
def get_monitors(tune_real, tune_imag, n_turns, slicer, Q_x, Q_y):
    filename_appdx='(dQreal={0:.3f},dQimag={1:.3f}'.format(tune_real*1e3, tune_imag*1e3)
    filename = '/home/vgubaidulin/Documents/PhD/Data_server/octupoles/'
    bunch_filename = filename+'bunch_mon'+filename_appdx
    slice_filename = filename+'slice_mon'+filename_appdx
    particle_filename = filename+'particle_mon'+filename_appdx
    parameters_dict = {'Q_x': Q_x, 'Q_y': Q_y, 'turns': n_turns, 'macroparticles': n_macroparticles}
    bunch_monitor = BunchMonitor(
        filename=bunch_filename, n_steps=n_turns, 
        parameters_dict=parameters_dict, write_buffer_every=20
        )
    slice_monitor = SliceMonitor(filename=slice_filename, n_steps=2048, slicer=slicer)
    particle_monitor = ParticleMonitor(filename=particle_filename, n_steps=n_turns, stride=10)
    return bunch_monitor, slice_monitor, particle_monitor
def optimize_number_of_turns(dQcoh_imag):
    if dQcoh_imag != 0: 
        n_turns = int(10*1/(2*np.pi*dQcoh_imag))
    else:
        n_turns = 16384
    return n_turns
def run(r, i):
    n_turns = optimize_number_of_turns(i)
    phase =  270+np.arctan2(r, i)/(2*np.pi)*360
    tune = np.sqrt(r**2+i**2)*1e3
    if tune==0:
        dampingrate=0
    else:
        dampingrate = 1e3/(2*np.pi*tune)
    np.random.seed(42)
    bunch = generators.generate_Gaussian6DTwiss(
        macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
        gamma=gamma, mass=m_p, circumference=C,
        alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
        alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z)
    # ### Initial offset
    # bunch.xp += 1e-6
    # bunch.yp += 1e-6
    # ### ==============

    I_oct=550
    ampl_det = AmplitudeDetuning.from_octupole_currents_LHC(i_focusing=-I_oct, i_defocusing=I_oct)
    # app_x = bunch.p0*1.381/2e-6
    # app_y = bunch.p0*1.441/2e-6
    # app_xy = bunch.p0*0.981/2e-6
    # print(bunch.p0)
    # ampl_det = AmplitudeDetuning(app_x, app_y, app_xy)
    print(ampl_det.app_x, ampl_det.app_y, ampl_det.app_xy)
    damperx = TransverseDamper(dampingrate_x=dampingrate, dampingrate_y=None, phase=phase, local_beta_function=beta_x_inj, verbose=False)
    dampery = TransverseDamper(dampingrate_x = None, dampingrate_y = dampingrate, phase=phase, local_beta_function=beta_y_inj, verbose=False)
    chroma = Chromaticity(Qp_x=[0], Qp_y=[0])
    slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    trans_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, [ampl_det, chroma])
    trans_one_turn = [m for m in trans_map]
    map_ = trans_one_turn  + [long_map, damperx, dampery]
    bunch_monitor, slice_monitor, particle_monitor = get_monitors(r, i, n_turns, slicer, Q_x, Q_y)
    for i in range(n_turns): 
        for m in map_:
            m.track(bunch)
        bunch_monitor.dump(bunch)
        particle_monitor.dump(bunch)
        if n_turns - i < 2049:
            slice_monitor.dump(bunch)    

if __name__=='__main__':

    #Machine parameters declaration
    n_segments = 1
    Ekin = 7e12
    intensity = 1.1e11
    C = 26658.883
    pipe_radius = 5e-2

    Q_x = 64.28
    Q_y = 59.31
    Q_s = 0.0020443

    alpha_0 = [0.0003225]
    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj =  92.7
    beta_y_inj = 93.2

    sigma_z = 0.059958
    epsn_x = 2e-6 # [m rad]
    epsn_y = 2e-6 # [m rad]
    
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
    n_slices=50
    alpha_x = alpha_x_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)
    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_y = np.zeros(n_segments)

    filename = '/home/vgubaidulin/Documents/PhD/Data_server/octupoles/'
    dQcoh_real = 1e-3*np.linspace(-0.5, 0.5, 11)
    dQcoh_imag = 1e-3*np.linspace(0, 0.2, 11)
    np.save(filename+'dQcoh_real', dQcoh_real)
    np.save(filename+'dQcoh_imag', dQcoh_imag)
    np.set_printoptions(precision=3)  

    fig, (ax1, ax2) = plt.subplots(1, 2)
    points = []
    for r in dQcoh_real:
        for i in dQcoh_imag:
            points.append((r, i))
    run(0, 0)
    # Parallel(n_jobs=-1)(delayed(run)(r, i) for (r, i) in points)
