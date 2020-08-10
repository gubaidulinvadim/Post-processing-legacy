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
from PyHEADTAIL.elens.elens import ElectronLens
from PyHEADTAIL.spacecharge.spacecharge import TransverseGaussianSpaceCharge, TransverseLinearSpaceCharge
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.impedances import wakes, wake_kicks
from PyHEADTAIL.aperture.aperture import CircularApertureXY
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles import slicing
from PyHEADTAIL.rfq.rfq import RFQTransverseKick
from tqdm import tqdm

from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf

#Simulation parameters declaration
n_turns = 8192
n_turns_slice_monitor = 2048
n_macroparticles = 16384
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
    filename_appdx='(dQreal={0:.3f},dQimag={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    filename = '/home/vgubaidulin/PhD/Data/Stability_scans/sc_test6/'
    bunch_filename = filename+'BM'+filename_appdx
    parameters_dict = {'Q_x': Q_x, 'Q_y': Q_y, 'turns': n_turns, 'macroparticles': n_macroparticles}
    bunch_monitor = BunchMonitor(
        filename=bunch_filename, n_steps=n_turns, 
        parameters_dict=parameters_dict, write_buffer_every=20
        )
    # slice_monitor = SliceMonitor(filename=slice_filename, n_steps=n_turns_slice_monitor, slicer=slicer)
    # particle_monitor = ParticleMonitor(filename=particle_filename, n_steps=n_turns, stride=10)
    return bunch_monitor
def get_slice_monitor(tune_real, tune_imag, slicer, n_turns, Q_x, Q_y):
    folder = '/home/vgubaidulin/PhD/Data/Stability_scans/sc_test6/'
    filename = folder+'SLM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    slice_monitor = SliceMonitor(filename=filename, slicer=slicer, write_buffer_every=100, n_steps=n_turns)
    return slice_monitor
def get_particle_monitor(tune_real, tune_imag, n_turns, Q_x, Q_y):
    folder = '/home/vgubaidulin/PhD/Data/Stability_scans/sc_test6/'
    filename = folder+'PM(dQre={0:.3f},dQim={1:.3f})'.format(tune_real*1e3, tune_imag*1e3)
    parameters_dict = {'Q_x': Q_x, 'Q_y': Q_y, 'turns': n_turns}
    particle_monitor = ParticleMonitor(filename=filename, n_steps=n_turns,
    stride=10, parameters_dict=parameters_dict, write_buffer_every=100)
    return particle_monitor

def run(r, i):
    n_turns = 16384
    phase =  270+np.arctan2(r, i)/(2*np.pi)*360
    tune = np.sqrt(r**2+i**2)*1e3
    if tune==0:
        dampingrate=0
    else:
        dampingrate = 1e3/(2*np.pi*tune)
    np.random.seed(42)
    C = 26658.883
    bunch = generators.generate_Gaussian6DTwiss(
        macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
        gamma=gamma, mass=m_p, circumference=C,
        alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
        alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z,  limit_n_rms_x=3.5, limit_n_rms_y=3.5)
    slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
    I_oct=0
    ampl_det = AmplitudeDetuning.from_octupole_currents_LHC(i_focusing=-I_oct, i_defocusing=I_oct)
    chroma = Chromaticity(Qp_x=[0], Qp_y=[0])

    damperx = TransverseDamper(dampingrate_x=dampingrate, dampingrate_y=None, phase=phase, local_beta_function=beta_x_inj, verbose=False)
    dampery = TransverseDamper(dampingrate_x = None, dampingrate_y = dampingrate, phase=phase, local_beta_function=beta_y_inj, verbose=False)
    trans_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, [ampl_det])
    trans_one_turn = [m for m in trans_map]
    ###########################
    #Electron lens declaration 
    ###########################
    # oneslice = slicing.UniformBinSlicer(n_slices=1, n_sigma_z=4)
    # L_e = 2
    # # I_e = 0.25*np.ones((n_slices,), dtype=np.float64)
    # sigma_e = 1.0*bunch.sigma_x()
    # Ue = 10e3
    # gamma_e = 1 + Ue * e / (m_e * c**2)
    # beta_e = np.sqrt(1 - gamma**-2)
    # dQ_max = 0.001
    # elens = ElectronLens.RoundDCElectronLens(L_e, dQ_max, 1.0, beta_e, 'KV', bunch)
    C = 26658.883
    L_e = 2
    sigma_e = 4*bunch.sigma_x()
    Ue = 10e3
    gamma_e = 1 + Ue * e / (m_e * c**2)
    beta_e = np.sqrt(1 - gamma**-2)
    dQ_max = 0.001
    z = np.linspace(-4*bunch.sigma_z(), 4*bunch.sigma_z(), n_slices)
    I_e = 5*np.exp(-z**2/(2*bunch.sigma_z()**2))
    pelens = ElectronLens(L_e, I_e, sigma_e, sigma_e, beta_e, dist='KV')

    sc1 = TransverseLinearSpaceCharge(slicer, C)
    sc2 = TransverseGaussianSpaceCharge(slicer, C)
    ###########################
    ###########################
    # rfq = RFQTransverseKick(v_2=2e9, omega=800e6*2.*np.pi, phi_0=0.)
    
    map_ = trans_one_turn  + [long_map, damperx, dampery, sc1]
    bunch_monitor = get_monitors(r, i, n_turns, slicer, Q_x, Q_y)
    particle_monitor = get_particle_monitor(r, i, n_turns, Q_x, Q_y)
    slice_monitor = get_slice_monitor(r, i, slicer, n_turns, Q_x, Q_y)
    for i in range(n_turns):             
        for m in map_:
            m.track(bunch)
        bunch_monitor.dump(bunch)
        particle_monitor.dump(bunch)
        # slice_monitor.dump(bunch)
        
if __name__=='__main__':

    #Machine parameters declaration
    n_segments = 1
    Ekin = 7e12
    intensity = 1.1e11*25
    C = 26658.883
    pipe_radius = 5e-2

    Q_x = 63.31
    Q_y = 63.32
    Q_s = 0.0020443

    alpha_0 = [0.0003225]
    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj =  92.7
    beta_y_inj = 93.2

    sigma_z = 0.059958
    epsn_x = 2.5e-6 # [m rad]
    epsn_y = 2.5e-6 # [m rad]
    
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


    # filename = '/home/vgubaidulin/PhD/Data/Stability_scans/sc_test2/'
    # dQcoh_real = 1e-3*np.linspace(-2.0, 0.6, 11)
    # dQcoh_imag = 1e-3*np.linspace(0.0, 0.2, 21)
    # np.save(filename+'dQcoh_real', dQcoh_real)
    # np.save(filename+'dQcoh_imag', dQcoh_imag)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # points = []
    # for r in dQcoh_real:
        # for i in dQcoh_imag:
            # points.append((r, i))
    
    # Parallel(n_jobs=6)(delayed(run)(r, i) for (r, i) in tqdm(points))  
    run(0., 0.)