import numpy as np
import pyTMD.math
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# latitude and longitude
dlon, dlat = 0.625, 0.5
lat = np.arange(-90 + dlat/2.0, 90 + dlat/2.0, dlat)
lon = np.arange(0 + dlon/2.0, 360 + dlon/2.0, dlon)
gridlon, gridlat = np.meshgrid(lon, lat)
# colatitude and longitude in radians
theta = xr.DataArray(np.radians(90.0 - gridlat), dims=('y','x'))
phi = xr.DataArray(np.radians(gridlon), dims=('y','x'))

# minimum and maximum degree of spherical harmonics
lmin, lmax = (3, 4)
# number of rows and columns for subplots
nrows = (lmax - lmin) + 1
ncols = lmax + 1
# projection for the plots
projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)

# plot spherical harmonics
fig = plt.figure(num=1, figsize=(10,4.75))
for n, l in enumerate(range(lmin, lmax+1)):
    for tau in range(0, l+1):
        # setup subplot
        i = n*ncols + tau + 1
        ax = fig.add_subplot(nrows, ncols, i, projection=projection)
        # calculate spherical harmonics (and derivatives w.r.t. colatitude)
        Y_lm, dY_lm = pyTMD.math.sph_harm(l, theta, phi, m=tau)
        # plot the surface
        ax.pcolormesh(lon, lat, Y_lm.real,
            transform=ccrs.PlateCarree(),
            cmap='viridis', rasterized=True)
        # set the title
        ax.set_title(f'$l={l}, m={tau}$')
        # add coastlines and set global
        ax.coastlines()
        ax.set_global()
        # turn off the axis
        ax.set_axis_off()

# adjust spacing and show
plt.tight_layout()
plt.show()