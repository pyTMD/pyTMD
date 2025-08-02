import numpy as np
import pyTMD.arguments
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# create figure and subplots
fig = plt.figure(num=1, figsize=(13,5))
subfig = fig.subfigures(2, 1, hspace=0.05, height_ratios=(1.0, 2.0))
ax1 = subfig[0].subplots(ncols=1)
ax2 = subfig[1].subplots(ncols=3, sharey='row')
# ax2[0].sharey(ax1)
# set x and y limits
ax1.set_xlim(-0.06, 2.14)
ax1.set_ylim(1e-3, 2e2)
ax2[0].set_ylim(1e-3, 2e2)

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

# run for degree-2 constituents
l = 2
# parse the Cartwright and Edden (1973) table for degree-2 constituents
CTE = pyTMD.arguments._parse_tide_potential_table(
    pyTMD.arguments._ce1973_table_1,
    format='Cartwright')
# for each spectral line
for i, line in enumerate(CTE):
    # spherical harmonic dependence (order)
    TAU = line['tau']
    # Doodson coefficients for constituent
    S = line['s']
    H = line['h']
    P = line['p']
    # convert N for ascending lunar node
    N = -1.0*line['n']
    PP = line['pp']
    # use cosines for (m + n) even
    # and sines for (m + n) odd
    K = -1.0*np.mod(l + TAU, 2)
    # determine constituent phase using equilibrium arguments
    arguments = np.array([TAU, S, H, P, N, PP, K], dtype=np.float64)
    # calculate the angular frequency
    omega = pyTMD.arguments._frequency(arguments, method='ASTRO5')
    # skip z0 (permanent tide)
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

# run for degree-3 constituents
l = 3
# parse the Cartwright and Tayler (1971) table for degree-3 constituents
CTE = pyTMD.arguments._parse_tide_potential_table(
    pyTMD.arguments._ct1971_table_5,
    format='Cartwright')
# for each line in the table
for i, line in enumerate(CTE):
    # spherical harmonic dependence (order)
    TAU = line['tau']
    # Doodson coefficients for constituent
    S = line['s']
    H = line['h']
    P = line['p']
    # convert N for ascending lunar node
    N = -1.0*line['n']
    PP = line['pp']
    # use cosines for (l + tau) even
    # and sines for (l + tau) odd
    K = -1.0*np.mod(l + TAU, 2)
    # determine constituent phase using equilibrium arguments
    arguments = np.array([TAU, S, H, P, N, PP, K], dtype=np.float64)
    # calculate the angular frequency
    omega = pyTMD.arguments._frequency(arguments, method='ASTRO5')
    # convert to frequency (solar days per cycle)
    f = np.abs(omega*86400.0)/(2.0*np.pi)
    # amplitude in cm
    amp = 100.0*np.abs(line['Hs3'])
    # plot amplitudes
    ax1.semilogy([f, f], [0.0, amp], color='0.4', zorder=1)
    for j, fr in enumerate(frange):
        if (f >= fr[0]) and (f <= fr[1]):
            ax2[j].semilogy([f, f], [0.0, amp], color='0.4', zorder=1)

# create inset axes and set ticks
plot_colors = ['k', 'k', 'k']
labels = ['Long-Period', 'Diurnal', 'Semi-Diurnal']
connector_visibility = [False, True, False, True]
for i, ax in enumerate(ax2):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    inset_rectangle, inset_connectors = ax1.indicate_inset(
        bounds=(xmin, ymin, xmax-xmin, ymax-ymin),
        inset_ax=ax, facecolor=plot_colors[i], alpha=0.15,
        edgecolor=plot_colors[i], zorder=0)
    # set visibility of connectors
    for j, vis in enumerate(connector_visibility):
        inset_connectors[j].set_visible(vis)
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
ax1.set_ylabel('Amplitude [cm]', fontsize=10)
ax2[0].set_ylabel('Amplitude [cm]', fontsize=10)
ax2[1].set_xlabel('Frequency [cpd]', fontsize=10)
# set titles
ax1.set_title(f'Tidal Spectra', fontsize=12)
# adjust subplots
subfig[0].subplots_adjust(left=0.048,right=0.9975,bottom=0.0,top=0.85)
subfig[1].subplots_adjust(left=0.048,right=0.9975,bottom=0.12,top=0.975,wspace=0.05)
plt.show()
