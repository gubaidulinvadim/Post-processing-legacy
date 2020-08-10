from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py as hp
import PyNAFF as pnf
from matplotlib import pyplot as plt
from matplotlib import animation, rc
import seaborn as sbs
from FITX import fit_risetime
sbs.set(rc={'figure.figsize': (8.3, 5.2)}, style='darkgrid',
        palette='colorblind', context='notebook')

def set_rc():
    sbs.set(rc={'figure.figsize':(8.3,5.2),
            'text.usetex':True,
           'font.family':'serif',
           'font.size':20,
           'axes.linewidth':2,
           'lines.linewidth':3,
           'legend.fontsize':16,
           'legend.numpoints':1,},
        style='white',
        palette='colorblind',
        context='talk')
    return 0

def get_bunch_data(real, imag, folder, is_plane_x=True):
    symbol = 'x' if is_plane_x else 'y'

    bunch_filename = folder+'BM(dQre={0:.3f},dQim={1:.3f}).h5'.format(real*1e3, imag*1e3)
    bunch_file = hp.File(bunch_filename, 'r')
    mean_x = bunch_file['Bunch']['mean_'+symbol][:]
    mean_xp = bunch_file['Bunch']['mean_'+symbol+'p'][:]
    epsn_x = bunch_file['Bunch']['epsn_'+symbol][:]
    sigma_x = bunch_file['Bunch']['sigma_'+symbol][:]
    n_turns = bunch_file['Bunch']['mean_'+symbol].shape[0]
    bunch_file.close()
    return mean_x, mean_xp, epsn_x, sigma_x, n_turns


def get_slice_data(real, imag, folder, n_macroparticles=16384, is_plane_x=True):
    symbol = 'x' if is_plane_x else 'y'

    filename = folder + \
        'SLM(dQre={0:.3f},dQim={1:.3f}).h5'.format(real*1e3, imag*1e3)
    slice_file = hp.File(filename)
    density = slice_file['Slices']['n_macroparticles_per_slice'][:] / \
        n_macroparticles
    mean_x = slice_file['Slices']['mean_'+symbol][:]*density
    mean_z = slice_file['Slices']['mean_z'][:]
    slice_file.close()
    return mean_x, mean_z


def get_particles_data(real, imag, folder, is_plane_x=True):
    symbol = 'x' if is_plane_x else 'y'
    filename = folder + \
        'PM(dQre={0:.3f},dQim={1:.3f}).h5part'.format(real*1e3, imag*1e3)
    particle_file = hp.File(filename, 'r')
    n_particles = particle_file['Step#0'][symbol][:].shape[0]
    n_steps = len(particle_file.keys())
    x = np.empty((n_particles,  n_steps), dtype=np.float64)
    xp = np.empty((n_particles, n_steps), dtype=np.float64)
    for step in range(0, n_steps):
        x[:, step] = particle_file['Step#{0:.0f}'.format(step)][symbol][:]
        xp[:, step] = particle_file['Step#{0:.0f}'.format(step)][symbol+'p'][:]
    particle_file.close()
    return x, xp

def get_phase_space_evolution(x, xp):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=1800)
    fig, ax = plt.subplots()
    set_rc()
    line, = ax.plot([], [], marker='.', ls='')
    # initialization function: plot the background of each frame
    def init():
        ax.set_ylim(-10, 10)
        ax.set_xlim(-1.0, 1.0)
        X = x[:,0]/1e-3
        Y = xp[:,0]/1e-6
        line.set_data(X, Y)
        return (line,)
    def animate(i):
        line.set_data(x[:,i]/1e-3, xp[:,i]/1e-6)
        plt.legend((r'Turn={:4d}'.format(i),))
        return (line, )
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=range(0, 32000, 100), interval=500, blit=True)
    anim.save('/home/vgubaidulin/PhD/Results/Unsorted/phase_space_temp.mp4', writer=writer)


def get_tunes(x, y):
    n_particles = x.shape[0]
    Q_x = np.empty((n_particles,), dtype=np.float64)
    Q_y = np.empty((n_particles,), dtype=np.float64)
    for p in range(n_particles):
        Q_x[p] = pnf.naff(x[p, :], turns=500)[:, 1]
        Q_y[p] = pnf.naff(y[p, :], turns=500)[:, 1]
    return Q_x, Q_y


def is_stable(mean_x, mean_xp, epsn_x, beta_x, n_turns):
    assert mean_x.shape[0] > 1000, ('Less than 1000 elements in mean_x')
    assert mean_xp.shape[0] > 1000, ('Less than 1000 elements in mean_xp')
    assert epsn_x.shape[0] > 1000, ('Less than 1000 elements in epsn_x')
    is_stable_x = True

    signal_x = np.sqrt(mean_x**2+(beta_x*mean_xp)**2)
    smoothing_window_size = 256
    min_level = 10 * np.max(signal_x[:1000])

    rx = fit_risetime(
        signal_x, min_level=min_level,
        smoothing_window_size=smoothing_window_size, 
        start_from_0=True
    )

    if np.isnan(rx) or rx < 0:
        is_stable_x = False
    elif np.max(epsn_x) > 1.10*np.max(epsn_x[:1000]):
        is_stable_x = False
    elif np.max(mean_x) > 10*np.max(mean_x[:1000]):
        is_stable_x = False

    return is_stable_x


def read_scan_data(folder):
    dQcoh_re = np.load(folder+'dQcoh_real.npy')
    dQcoh_im = np.load(folder+'dQcoh_imag.npy')

    growth_rates = np.empty(
        shape=(len(dQcoh_re), len(dQcoh_im)), dtype=np.float64)
    stable_points = np.empty(
        shape=(len(dQcoh_re), len(dQcoh_im)), dtype=np.bool_)
    growth_rates.fill(np.nan)
    stable_points.fill(False)

    for r, index_re in enumerate(dQcoh_re):
        for i, index_im in enumerate(dQcoh_im):
            mean_x, mean_xp, epsn_x, beta_x, n_turns = get_bunch_data_x(
                r, i, folder)
            if is_stable(mean_x, mean_xp, epsn_x, beta_x, n_turns):
                stable_points[index_re, index_im] = is_stable(
                    mean_x, mean_xp, epsn_x, beta_x, n_turns)
                signal_x = np.sqrt(mean_x**2+(beta_x*mean_xp)**2)
                smoothing_window_size = 256
                min_level = 10 * np.max(signal_x[:1000])

                rx = fit_risetime(
                    signal_x, min_level=min_level,
                    smoothing_window_size=smoothing_window_size
                )
                growth_rates[index_re, index_im] = rx
    return stable_points, growth_rates

#TODO: rewrite newly added functionality
def get_growth_rates(dQcoh_real, dQcoh_imag, folder, beta_x=93.7, beta_y=92.6):
    growth_rates = np.zeros(shape=(len(dQcoh_real), len(dQcoh_imag)), dtype=np.float64)
    tunes = np.zeros(shape=(len(dQcoh_real), len(dQcoh_imag)), dtype=np.float64)
    for r_index, r in enumerate(dQcoh_real):
        for i_index, i in enumerate(dQcoh_imag):
            mean_x, mean_xp, epsn_x, sigma_x, n_turns = get_bunch_data(r, i, folder)
            signal_x = 1e6*np.sqrt((mean_x)**2 + (beta_x * mean_xp)**2)
            index = np.where(signal_x < 1e8)[0][-1]
            signal_x = signal_x[:index]
            smoothing_window_size = 512
            rx = fit_risetime(
                signal_x, min_level=min_level, 
                smoothing_window_size=smoothing_window_size,
                matplotlib_axis=None, 
                start_from_0=True
            )
            if ( (not np.isnan(rx)) and rx > 0):
                growth_rates[r_index, i_index] = 1/(2*np.pi*rx)
                tunes[r_index, i_index] = i
            else:
                growth_rates[r_index, i_index] = 0
                tunes[r_index, i_index] = i
    return growth_rates, tunes

def get_stability_point(tunes, growth_rates):
    x = np.array(tunes).reshape(-1, 1)/1e-3
    y = np.array(growth_rates)/1e-3
    try:
        ransac = RANSACRegressor()
        ransac.fit(x, y)
        predicted_y = ransac.predict(x)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        point = -ransac.estimator_.intercept_/ransac.estimator_.coef_[0]
        stability_boundary = point if point > 0 else 0
        r2 = r2_score(ransac.predict(x[inlier_mask]), y[inlier_mask])
    except:
        stability_boundary = None
        r2 = -42
    if r2 < 0.75 or ransac.estimator_.coef_[0] < 0:
        stability_boundary = None
    return stability_boundary, r2