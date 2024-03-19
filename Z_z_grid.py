"""Computes the minimum Jeans mass on a grid of redshifts and metallicities"""

from itertools import product
from masses import *

Zgrid = [0.01, 0.1, 1, 3]
zgrid = np.array([0, 5, 7, 10])
params = np.array(list(product(zgrid, Zgrid)))
blackbody_temp = (1 + params[:, 0]) * 2.73

mjmin = minimum_jeans_mass(blackbody_temp=blackbody_temp, Zd=params[:, 1])
mlarson = larson_mass(blackbody_temp=blackbody_temp, Zd=params[:, 1])

np.savetxt(
    "Z_z_grid.csv",
    np.c_[params[:, 0], params[:, 1], mjmin, mlarson],
    delimiter=", ",
    header="# (0) Metallicity Z (1) Redshift z (2) Min. Jeans mass (3) Hydrostatic core mass",
)
