import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
# import tide programs
import pyTMD.astro
import pyTMD.arguments

def frequency(arguments):
    """
    Calculates the angular frequencies of constituents
    """
    # Modified Julian Dates at J2000
    MJD = np.array([51544.5, 51544.55])
    # time interval in seconds
    deltat = 86400.0*(MJD[1] - MJD[0])
    # calculate the mean longitudes of the sun and moon
    s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD, ASTRO5=True)
    # initial time conversions
    hour = 24.0*np.mod(MJD, 1)
    # convert from hours solar time into mean lunar time in degrees
    tau = 15.0*hour - s + h
    # determine equilibrium arguments
    fargs = np.c_[tau, s, h, p, n, pp]
    rates = (fargs[1,:] - fargs[0,:])/deltat
    fd = np.dot(rates, arguments)
    # convert to radians per second
    omega = 2.0*np.pi*fd/360.0
    return omega

# Cartwright and Edden (1973) table with updated values
table = pyTMD.arguments._ce1973_table_1
# read the table
CTE = pyTMD.arguments._parse_tide_potential_table(table)

# create figure and subplots
fig = plt.figure(num=1, figsize=(13,5))
subfig = fig.subfigures(2, 1, hspace=0.05)
ax1 = subfig[0].subplots(ncols=1)
ax2 = subfig[1].subplots(ncols=3, sharey='row')
ax2[0].sharey(ax1)
# set x and y limits of the main plot
ax1.set_xlim(-0.1, 2.18)
ax1.set_ylim(1e-3, 3e2)

# major constituents to label for each species
major = []
major.append(['mm', 'mf', 'mtm'])
major.append(['q1', 'o1', 'k1', 'j1'])
major.append(['2n2', 'm2', 'l2', 's2', 'n2'])
# frequency ranges for each species band
frange = []
frange.append([0, 0.5])
frange.append([0.80, 1.15])
frange.append([1.75, 2.10])
# for each spectral line
for i, line in enumerate(CTE):
    # calculate the angular frequency
    arguments = np.array([line[c] for c in ['tau','s','h','p','n','pp']])
    omega = frequency(arguments)
    # skip z0
    if (omega == 0.0):
        continue
    # convert to frequency (solar days per cycle)
    f = np.abs(omega*86400.0)/(2.0*np.pi)
    # amplitude in cm
    amp = 100.0*np.abs(line['Hs3'])
    # get the constituent ID based on the first 6 arguments
    cons = pyTMD.arguments._to_constituent_id(arguments,
        arguments=6, raise_error=False)
    # plot amplitudes and color if in the major constituents list
    ax1.semilogy([f, f], [0.0, amp], color='0.4', zorder=1)
    for j, fr in enumerate(frange):
        if (f >= fr[0]) and (f <= fr[1]) and (cons in major[j]):
            ax2[j].semilogy([f, f], [0.0, amp], color='red', zorder=2)
            ax2[j].text(f, 1.5*amp, cons, color='red', fontsize=10, ha='center')
            break
        elif (f >= fr[0]) and (f <= fr[1]):
            ax2[j].semilogy([f, f], [0.0, amp], color='0.4', zorder=1)
            break

# create inset axes and set ticks
plot_colors = ['mediumseagreen', 'darkorchid', 'darkorange']
labels = ['Long-Period', 'Diurnal', 'Semi-Diurnal']
for i, ax in enumerate(ax2):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax1.indicate_inset(bounds=(xmin, ymin, xmax-xmin, ymax-ymin),
        inset_ax=ax, facecolor=plot_colors[i], alpha=0.25,
        edgecolor=plot_colors[i], zorder=0)
    # add labels to inset axes
    prop = dict(size=12, weight='bold', color=plot_colors[i])
    at = offsetbox.AnchoredText(labels[i], loc=2, pad=0,
        borderpad=0.5, frameon=False, prop=prop)
    ax.axes.add_artist(at)
    # set ticks for inset axes
    ax.get_xaxis().set_tick_params(which='both', direction='in', color=plot_colors[i])
    ax.get_yaxis().set_tick_params(which='both', direction='in', color=plot_colors[i])
    # stronger linewidth on frame
    for key,val in ax.spines.items():
        val.set_linewidth(1.5)
        val.set_color(plot_colors[i])
# set ticks
ax1.get_xaxis().set_tick_params(which='both', direction='in')
ax1.get_yaxis().set_tick_params(which='both', direction='in')
[val.set_linewidth(1.5) for key,val in ax1.spines.items()]
# # add x and y labels
ax1.set_ylabel('Amplitude [cm]')
ax2[0].set_ylabel('Amplitude [cm]')
ax2[1].set_xlabel('Frequency [cpd]')
# set titles
ax1.set_title(f'Tidal Spectra')
# adjust subplots
subfig[0].subplots_adjust(left=0.05,right=0.995,bottom=0.0,top=0.90,wspace=0.05)
subfig[1].subplots_adjust(left=0.05,right=0.995,bottom=0.175,top=0.95,wspace=0.05)
plt.show()
