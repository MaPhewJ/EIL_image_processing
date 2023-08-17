import imageio as io 
import numpy as np
import numpy.ma as ma
from skimage.measure import regionprops, regionprops_table
from scipy.ndimage import distance_transform_edt as dist_trans
from tqdm import tqdm
import pandas as pd

# Graylevel Radial Analysis of Particles in ElectrodeS (GRAPES)

def GRAPES(labels_arr, grey_arr, normalised_by = 'radial_max'):
    props = regionprops_table(labels_arr, intensity_image = grey_arr, 
                          properties = ('label', 'area', 'centroid', 'axis_major_length', 'equivalent_diameter_area', 'intensity_max', 'intensity_mean', 'intensity_min', 'image', 'image_intensity'),
                          cache=True)
    props = pd.DataFrame(props)
    GRAPES_rp = []
    GRAPES_gl = []
    GRAPES_norm_gl = []
    for x, y in zip(props['image'], props['image_intensity']):
        rp, gl, norm_gl = _grapes_(x,y)
        GRAPES_rp.append(rp)
        GRAPES_gl.append(gl)
        GRAPES_norm_gl.append(norm_gl)
    props['GRAPES_RP'] = GRAPES_rp
    props['GRAPES_GL'] = GRAPES_gl
    props['GRAPES_GL_NORMED'] = GRAPES_norm_gl
    props = props.rename(columns={'area':'volume'})
    props = props.rename(columns={'equivalent_diameter_area':'equivalent_diameter_volume'})
    #props = props.drop(columns=['image','image_intensity'])
    return props


normalised_by = 'radial_max'

def _grapes_(region_mask, intensity_im):
    lbl = np.pad(region_mask, 1)
    intty = np.pad(intensity_im, 1)
    edt = np.round(dist_trans(lbl))
    norm_gl = []
    gl = []
    rp = []
    for radial_dst in range(1, int(np.amax(edt))):
        radial_mask = ma.masked_where(edt != radial_dst, intty)
        rp.append(radial_dst)
        gl.append(ma.mean(radial_mask))
        norm_gl.append(ma.mean(radial_mask))
    rp = np.array(rp)
    gl = np.array(gl)
    norm_gl = np.array(norm_gl)
    if normalised_by == 'radial_max':
        try:
            radial_max = np.amax(gl)
            gl = (gl/radial_max).astype(np.float32)
        except ValueError:
            gl = np.empty_like(gl)
    if normalised_by == 'surface':
        surface = gl[0]
        gl = (gl/surface).astype(np.float32)
    return rp, gl, norm_gl




