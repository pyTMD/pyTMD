import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pyTMD

# adjust style
facecolor = "#fcfcfc"
plt.rcParams["figure.facecolor"] = facecolor
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato"]


def time_formatter(hours):
    ampm = "am" if hours < 12 else "pm"
    hour = np.mod(np.floor(hours), 12)
    minute = 60.0 * np.mod(hours, 1)
    return f"{hour:02.0f}:{minute:02.0f}{ampm}"


t = np.linspace(0, 1, 1000)
hours = t * 24
# synthetic tide with a small diurnal
h = np.sin(3.86 * np.pi * t + 0.4) + 0.1 * np.cos(2.0 * np.pi * t)
height = xr.DataArray(h, dims=("time",), coords=dict(time=hours))
# differentiate to calculate high and low tides
high_peaks, low_peaks = height.tmd.find_peaks()
high_tides = height.where(high_peaks, drop=True)
low_tides = height.where(low_peaks, drop=True)

# setup figure and subplots
fig, ax = plt.subplots(num=1, figsize=(8, 3))
# bounding box style
bboxpad = dict(boxstyle="square,pad=0.2", ec=facecolor, fc=facecolor)
# plot mean sea level
ax.axhline(0, color="0.4", linewidth=0.7)
# plot heights
ax.plot(height.time, height, color="0.4")
ax.fill_between(height.time, height, color="0.4", alpha=0.1)
# add vertical lines and times of high tides
for p in high_tides:
    ax.axvline(
        p.time,
        color="red",
        linewidth=0.7,
        linestyle="--",
        dashes=(5, 5),
    )
    time = time_formatter(p.time)
    ax.text(
        p.time,
        p + 0.1,
        time,
        color="red",
        ha="center",
        va="bottom",
        bbox=bboxpad,
    )
# add vertical lines and times of low tides
for p in low_tides:
    ax.axvline(
        p.time,
        color="dodgerblue",
        linewidth=0.7,
        linestyle="--",
        dashes=(5, 5),
    )
    time = time_formatter(p.time)
    ax.text(
        p.time,
        p - 0.1,
        time,
        color="dodgerblue",
        ha="center",
        va="top",
        bbox=bboxpad,
    )
# format axes
ax.xaxis.set_major_locator(MultipleLocator(6))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_xlim(0, 24)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("Time [hour]")
ax.set_ylabel("Height [m]")
fig.tight_layout()
plt.show()
