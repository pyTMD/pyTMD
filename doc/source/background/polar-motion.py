import timescale
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# read IERS daily polar motion values
EOP = timescale.eop.iers_daily_EOP()
ts = timescale.time.Timescale(EOP['MJD'])
# calculate angular coordinates of mean/secular pole at time
mpx, mpy, fl = timescale.eop.iers_mean_pole(ts.year, convention='2018')
# calculate differentials from mean/secular pole positions
# using the latest definition from IERS Conventions (2010)
mx = EOP['x'] - mpx
my = -(EOP['y'] - mpy)

# create figure and subplots
fig = plt.figure(num=1, figsize=(8.5,4))
subfig = fig.subfigures(1, 2, wspace=0.05, width_ratios=(1.5, 1.0))
ax1 = subfig[0].subplots(nrows=2, sharex=True)
ax2 = subfig[1].subplots()

# create plot of earth orientation parameters
ax1[0].plot(ts.year, EOP['x'], color='0.4', label='IERS')
ax1[0].plot(ts.year, mpx, color='red', label='Secular')
ax1[1].plot(ts.year, EOP['y'], color='0.4', label='IERS')
ax1[1].plot(ts.year, mpy, color='red', label='Secular')
# set axis labels
ax1[0].set_ylabel('X Pole [asec]', fontsize=10, labelpad=0)
ax1[1].set_ylabel('Y Pole [asec]', fontsize=10, labelpad=8)
ax1[1].set_xlabel('Time [yr]', fontsize=10)
labels = ['a)', 'b)']
for i, label in enumerate(labels):
    ax1[i].tick_params(which='both', direction='in')
    at = offsetbox.AnchoredText(label,
        loc=2, pad=0.0, borderpad=0.5, frameon=False,
        prop=dict(size=12,weight='bold',color='k'))
    ax1[i].axes.add_artist(at)
# add legend
lgd = ax1[1].legend(frameon=False, labelspacing=0.1, borderpad=0.1)
for line in lgd.get_lines():
    line.set_linewidth(6)

# plot deviation from mean/secular pole
sc = ax2.scatter(mx, my, c=ts.year, cmap='plasma_r', s=0.5)
ax2.axhline(0, color='0.4', ls='--', lw=0.5)
ax2.axvline(0, color='0.4', ls='--', lw=0.5)
# add axis labels
ax2.set_xlabel('X Pole [asec]', fontsize=10)
ax2.set_ylabel('Y Pole [asec]', fontsize=10, labelpad=3)
labels = ['a)', 'b)']
at = offsetbox.AnchoredText('c)',
    loc=2, pad=0.0, borderpad=0.5, frameon=False,
    prop=dict(size=12,weight='bold',color='k'))
ax2.axes.add_artist(at)
# add title
ax2.set_title('Deviation from Secular Pole', fontsize=10)
# set axis limits
ax2.set_xlim([-0.35, 0.35])
ax2.set_ylim([-0.35, 0.35])
ax2.tick_params(which='both', direction='in')
ax2.set_aspect('equal')
ax2.set_aspect('equal')

# Add colorbar with a colorbar axis
# Add an ax2es at position rect [left, bottom, width, height]
cbar_ax = subfig[1].add_axes([0.12, 0.085, 0.87, 0.04])
# extend = add extension triangles to upper and lower bounds
# options: neither, both, min, max2
cbar = subfig[1].colorbar(sc, cax=cbar_ax, extend='neither',
    drawedges=False, orientation='horizontal')
# rasterized colorbar to remove lines
cbar.solids.set_rasterized(True)
# Add label to the colorbar
cbar.ax.set_title('Time [yr]', fontsize=10, rotation=0, y=-1.75, va='top')
cbar.ax.xaxis.set_label_coords(1.075, 0.5)
# set tick parameters
cbar.ax.tick_params(which='both', width=1, length=5, direction='in')

# adjust subplot within figure
subfig[0].subplots_adjust(left=0.11,right=0.99,bottom=0.1,top=0.99,hspace=0.1)
subfig[1].subplots_adjust(left=0.12,right=0.99,bottom=0.19,top=0.99)
plt.show()