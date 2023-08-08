import numpy as np
import imageio as io
import os
from tomopy.prep.normalize import normalize as flat_dark
from tomopy.recon.algorithm import recon
from tomopy.recon.rotation import write_center as CoR

path = 'path/to/projections/*.tiff'
proj_arr = []
for i in os.listdir(path):
    proj = io.imread(i)
    proj_arr.append(proj)
proj_arr = np.array(proj_arr)

angle = 0 
angles = []
for j in range(1216):
    angles.append(angle)
    angle += (360/1215)
angles = np.array(angles)
angles = angles * (np.pi/180)

flat = io.imread('path/to/flat.tiff')
dark = io.imread('path/to/dark.tiff')

proj_arr = flat_dark(proj_arr, flat, dark, 'mean')
io.volwrite('normed_proj.tiff', proj_arr, format='tiff', bigtiff=True)

centre = CoR(proj_arr, angles, cen_range = [(int(proj_arr.shape[0]/2)-50), (int(proj_arr.shape[0]/2)+50), 5], algorithm = 'fbp', filter_name = 'ramlak')
centre = 750

recon_vol = recon(proj_arr, angles, center = centre, algorithm = 'fbp', filter_name = 'ramlak')
io.volwrite('recon.tiff', recon_vol, format='tiff', bigtiff=True)


