import datetime
import zoneinfo
import numpy as np
import pyTMD.compute
import timescale.time
import matplotlib.pyplot as plt

# local timezone info
tzinfo = zoneinfo.ZoneInfo("US/Pacific")
# start and end times (local)
start = datetime.datetime(2026, 1, 1, tzinfo=tzinfo)
end = start + datetime.timedelta(days=10)
# create a range of dates converting to UTC
UTC = timescale.time.date_range(start.isoformat(), end.isoformat(), 1, "m")

# calculate a tide prediction for the local time range
model = "GOT4.10_nc"
lon, lat = -125.1, 46.9
tpred = pyTMD.compute.tide_elevations(
    lon, lat, UTC, model=model, standard="datetime", infer_minor=True
)

# local datetime array
local = UTC + np.timedelta64(start.utcoffset())
# assign the local time as the time coordinate
tpred = tpred.assign_coords(time=local)

# plot the tide predictions in local time
fig, ax = plt.subplots(facecolor="#fcfcfc")
tpred.plot(ax=ax)
ax.set_xlabel(f"Time [{tzinfo.key}]")
ax.set_ylabel(f"{model} Tide Elevation [m]")
ax.set_title(f"{lon:0.1f}\u00b0N {lat:0.1f}\u00b0E")
