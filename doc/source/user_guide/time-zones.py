import datetime
import zoneinfo
import numpy as np
import pyTMD.compute
import timescale.time
import matplotlib.pyplot as plt

# local timezone info
tzinfo = zoneinfo.ZoneInfo("America/Los_Angeles")
# start and end times (local)
start = datetime.datetime(2026, 1, 1, tzinfo=tzinfo)
end = start + datetime.timedelta(days=10)
# create a range of dates using timescale to convert to UTC
UTC = timescale.time.date_range(start.isoformat(), end.isoformat(), 1, "m")

# calculate a tide prediction using the UTC datetimes
model = "GOT4.10_nc"
lon, lat = -125.1, 46.9
tpred = pyTMD.compute.tide_elevations(
    lon, lat, UTC, model=model, standard="datetime", infer_minor=True
)

# NOTE: using a single UTC offset only works when the time range does not
# include a transition to daylight saving time or the local time zone
# does not observe DST. In the case where there is a transition to DST:
# the offset will need to be added in pieces (before and after transition)
utcoffset = np.timedelta64(start.utcoffset())
# calculate the local datetime array
local = UTC + utcoffset
# assign the local time as the DataArray time coordinate
tpred = tpred.assign_coords(time=local)

# plot the tide predictions in local time
fig, ax = plt.subplots(facecolor="#fcfcfc")
tpred.plot(ax=ax)
ax.set_xlabel(f"Time [{tzinfo.key}]")
ax.set_ylabel(f"{model} Tide Elevation [m]")
ax.set_title(f"{lon:0.1f}\u00b0N {lat:0.1f}\u00b0E")
