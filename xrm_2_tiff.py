import dxchange as dxc #need to make sure you olefile installed for dxc to work
import imageio as io
import numpy as np

vol, meta = dxc.reader.read_txm('D:/versa/SiC_anode_nc_80kv/recon.txm')
io.volwrite('path/to/write.tiff', vol, format='tiff', bigtiff=True)
