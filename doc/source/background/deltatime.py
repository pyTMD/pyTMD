import datetime
import timescale
import numpy as np
import matplotlib.pyplot as plt

# create a timescale object from a range of dates
start_date = np.array('1989-01-01', dtype=f'datetime64[D]')
end_date = np.array(datetime.datetime.now(), dtype=f'datetime64[D]')
ts = timescale.from_range(start_date, end_date)
# calculate TT-UT1 and convert to seconds
tt_ut1 = 86400.0*ts.tt_ut1
# get leap seconds
leaps = timescale.time.get_leap_seconds(truncate=False)
# truncate to dates after start of time series
leaps = leaps[leaps > ts.gps[0]]
n_leaps = len(leaps)

# create figure
fig, ax = plt.subplots(num=1)
ax.plot(ts.year, tt_ut1, '0.4')
plot_colors = iter(plt.cm.rainbow(np.linspace(0, 1, n_leaps)))
for leap in leaps:
    ii, = np.nonzero(np.isclose(ts.gps, leap))
    cal = ts[ii].to_calendar()
    label = f'{cal.year[0]:4.0f}-{cal.month[0]:02.0f}-{cal.day[0]:02.0f}'
    ax.plot(ts.year[ii], tt_ut1[ii], '-*', lw=0,
        color=next(plot_colors), label=label)
# set axis labels
ax.set_ylabel('TT-UT1 [s]')
ax.set_xlabel('Time [yr]')
ax.tick_params(which='both', direction='in')
# add legend
lgd = ax.legend(loc=4, frameon=False, title='Leap Seconds')
for line in lgd.get_lines():
    line.set_linewidth(6)
fig.tight_layout()
plt.show()
