import numpy as np
import pyTMD.constituents
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# number of periods
N = 5000
# calculate over diurnal range
periods = np.linspace(0.85, 1.15, N)
# convert to radians per second
omegas = 2.0 * np.pi / (86400.0*periods)
# 1066A-N values from Wahr (1979)
h0, k0, l0 = np.array([6.03e-1, 2.98e-1, 8.42e-2])
# initialize arrays
love = {}
for i, key in enumerate(['h', 'k', 'l']):
    love[key] = np.zeros((N))
# calculate Love numbers
for i, omega in enumerate(omegas):
    love['h'][i], love['k'][i], love['l'][i] = \
        pyTMD.constituents._love_numbers(omega, model='1066A-N')

# create figure and subplots
fig, ax = plt.subplots(num=1, nrows=3, sharex=True, figsize=(6, 5))
# plot Love numbers
for i, key in enumerate(['h', 'k', 'l']):
    # remove the largest gradient
    grad = np.gradient(love[key])
    ii, = np.nonzero(np.abs(grad) == np.abs(grad).max())
    love[key][ii] = np.nan
    # plot Love numbers
    ax[i].plot(periods, love[key], '0.4')

# add markers for individual constituents
cons = ['o1', 'p1', 'k1', 'phi1', 'j1',  'oo1']
labels= ['o1', 'p1', 'k1', '\u03C61', 'j1',  'oo1']
plot_colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(cons))))
for i, c in enumerate(cons):
    om = pyTMD.constituents.frequency(c)
    p = 2.0 * np.pi / (86400.0*om)
    h, k, l = pyTMD.constituents._love_numbers(om, model='1066A-N')
    s, = ax[0].plot(p, h, '.', color=next(plot_colors), label=labels[i])
    ax[1].plot(p, k, '.', color=s.get_markerfacecolor(), label=labels[i])
    ax[2].plot(p, l, '.', color=s.get_markerfacecolor(), label=labels[i])

# adjust axes
ax[0].set_xlim(periods.max(), periods.min())
ax[0].set_ylim(h0 - 0.22, h0 + 0.22)
ax[1].set_ylim(k0 - 0.18, k0 + 0.18)
ax[2].set_ylim(l0 - 0.02, l0 + 0.02)

# set axis labels
ax[2].set_xlabel('Period [day]')
ax[0].set_ylabel('$h_2$', labelpad=10)
ax[1].set_ylabel('$k_2$', labelpad=10)
ax[2].set_ylabel('$l_2$')
labels = ['a)', 'b)', 'c)']
for i, label in enumerate(labels):
    ax[i].tick_params(which='both', direction='in')
    at = offsetbox.AnchoredText(label,
        loc=2, pad=0.0, borderpad=0.5, frameon=False,
        prop=dict(size=12,weight='bold',color='k'))
    ax[i].axes.add_artist(at)

# add legend
lgd = ax[2].legend(loc=4, frameon=False, ncols=2, labelspacing=0.2, borderpad=0.05)
for line in lgd.get_lines():
    line.set_markersize(10.0)

# adjust subplots
fig.subplots_adjust(top=0.99, bottom=0.085, left=0.10, right=0.95, hspace=0.1)
plt.show()
