import datetime
import timescale
import numpy as np
import matplotlib.pyplot as plt

# create a timescale object from a range of dates
start_date = np.array('1990-01-01', dtype=f'datetime64[D]')
end_date = np.array(datetime.datetime.now(), dtype=f'datetime64[D]')
ts = timescale.from_range(start_date, end_date)
# calculate TT-UT1 and convert to seconds
tt_ut1 = 86400.0*ts.tt_ut1

fig, ax = plt.subplots(num=1)
ax.plot(ts.year, tt_ut1)
ax.set_ylabel('TT-UT1 [s]')
ax.set_xlabel('Time [yr]')
fig.tight_layout()
plt.show()
