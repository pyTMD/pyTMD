import datetime
import timescale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# create a timescale object from a range of dates
start_date = np.array('1980-01-01', dtype=f'datetime64[D]')
end_date = np.array(datetime.datetime.now(), dtype=f'datetime64[D]')
ts = timescale.from_range(start_date, end_date)
# calculate TT-UT1 and convert to seconds
tt_ut1 = 86400.0*ts.tt_ut1
# get leap seconds
leaps = timescale.time.get_leap_seconds(truncate=False)
# truncate to dates after start of time series
leaps = leaps[leaps > ts.gps[0]]
n_leaps = len(leaps)

# dynamic time is ahead of TAI by 32.184 seconds
tt_tai = 32.184
# TAI time is ahead of GPS by 19 seconds
tai_gps = 19.0
# convert from dynamic time to TAI
TAI = np.atleast_1d(ts.tt - 2444244.5)*ts.day - tt_tai
gps_utc = timescale.time.count_leap_seconds(TAI - tai_gps)
# recalculate UT1-UTC (seconds)
ut1_utc = tt_tai + tai_gps + gps_utc - tt_ut1

# create figure
fig, ax = plt.subplots(num=1, nrows=2, sharex=True, figsize=(6, 6))
# plot UT1-UTC and TT-UT1
ax[0].plot(ts.year, ts.ut1_utc, '.', ms=0.5, c='0.4')
ax[1].plot(ts.year, tt_ut1, '0.4')
# plot leap seconds
plot_colors = iter(plt.cm.rainbow(np.linspace(0, 1, n_leaps)))
for leap in leaps:
    ii, = np.nonzero(np.isclose(ts.gps, leap))
    cal = ts[ii].to_calendar()
    label = f'{cal.year[0]:4.0f}-{cal.month[0]:02.0f}-{cal.day[0]:02.0f}'
    l, = ax[1].plot(ts.year[ii], tt_ut1[ii], '-*', lw=0,
        color=next(plot_colors), label=label)
    ax[0].axvline(ts.year[ii], color=l.get_color(), lw=0.5)
# set axis labels
ax[0].set_ylabel('UT1-UTC [s]', labelpad=2)
ax[1].set_ylabel('TT-UT1 [s]')
ax[1].set_xlabel('Time [yr]')
labels = ['a)', 'b)']
for i, label in enumerate(labels):
    ax[i].tick_params(which='both', direction='in')
    at = offsetbox.AnchoredText(label,
        loc=2, pad=0.0, borderpad=0.5, frameon=False,
        prop=dict(size=12,weight='bold',color='k'))
    ax[i].axes.add_artist(at)
# add legend
lgd = ax[1].legend(loc=1, frameon=False,
    title='Leap Seconds', title_fontproperties=dict(weight='normal'),
    bbox_to_anchor=(0.82, 0.0, 0.18, 1.0), bbox_transform=fig.transFigure)
for line in lgd.get_lines():
    line.set_linewidth(6)
# adjust subplots
fig.subplots_adjust(top=0.99, bottom=0.065, left=0.10, right=0.75, hspace=0.05)
plt.show()
