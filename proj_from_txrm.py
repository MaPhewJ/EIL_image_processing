import numpy as np
import cv2
import imageio as io
import dxchange as dxc

#In this script we extract usable projection from a txrm file 
#output by a zeiss x-ray microscope
#the script will translate, and apply flat correction

vol, meta = dxc.reader.read_txrm('si_c_anode_exsitu_24723_tomo-A.txrm') #read a txrm file

transform = []
for i,m in zip(meta['x-shifts'], meta['y-shifts']): #make a list of x-shift and y-shifts
    mytup = [i,m]
    transform.append(mytup)

ref_norm = meta['reference']/np.amax(meta['reference']) #normalise flat field image

vol1 = []
for i,m in zip(transform, vol): #Apply shifts and flat field correction to each projection
    translate = np.float32([
    [1,0,i[0]],
    [0,1,i[1]]
    ])
    im = cv2.warpAffine(m, translate, (m.shape[1], m.shape[0]))
    refn = cv2.warpAffine(ref_norm, translate, (ref_norm.shape[1], ref_norm.shape[0]))
    im = im/refn
    im = np.where(np.isnan(im)==True,0,im)
    im = ((im/np.amax(im))*65535).astype(np.uint16)
    vol1.append(im)

io.volwrite('projections.tiff', vol1, format='tiff', bigtiff=True)