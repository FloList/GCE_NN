import astropy
from astropy.io import fits

# Load 4FGL catalogue
filename = "/home/flo/Documents/Projects/GCE_hist/Sources/gll_psc_v31.fit"
hdul = fits.open(filename)
hdul.info()
fermi_lat_data = hdul[1].data
dat = astropy.table.Table.read(filename, format='fits', hdu=1)

# Load weekly data
filename_weekly = "/home/flo/Documents/Projects/GCE_hist/Sources/lat_photon_weekly_w766_p305_v001.fits"
hdul_weekly = fits.open(filename_weekly)
dat_weekly = astropy.table.Table.read(filename_weekly, format='fits', hdu=1)

# Load lightcurve data
filename_lightcurve = "/home/flo/Documents/Projects/GCE_hist/Sources/3FGL_J0000d1p6545_lc.fits"
hdul_lightcurve = fits.open(filename_lightcurve)
dat_lightcurve = astropy.table.Table.read(filename_lightcurve, format='fits', hdu=1)
