#Import Libraries 
import numpy as np
import imageio as io
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import concurrent.futures
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm

#Application of parralelised non-local means filter to images of electrode particles during mg33497-1 beamtime. 
#This will check if the image has already been processed before running and can therefore be run continuously 

def apply_nlm(image):
    img = img_as_float(image) ###load image to be smoothed by NLM.
    img = denoise_nl_means(img, h=0.00472*4.5, fast_mode=True, patch_size=10, patch_distance=15, channel_axis=None)  
    return (img*255).astype(np.uint8)
    
def main():
    processed = []
    for i in glob('/dls/i13/data/2023/mg33497-1/processing/8bit_tiffs_cropped_nlm/*.tiff'):
        processed.append(str(os.path.basename(Path(i)))[:6])
    for i in tqdm(glob('/dls/i13/data/2023/mg33497-1/processing/8bit_tiffs_cropped/*.tiff')):
        if str(os.path.basename(Path(i)))[:6] in processed:
            pass
        elif int(os.path.basename(Path(i))[:6]) < 162949:
            pass
        else:
            print(f'processing {i}')
            vol = io.volread(i)
            shape = vol.shape
            arr = np.empty(shape, np.uint8)
            shape = list(vol.shape)
            image_files = []
            for l in range(shape[0]):###loops through tomo slices create image list
                image_files.append(vol[l])
            with concurrent.futures.ProcessPoolExecutor() as executor:
                iterable = executor.map(apply_nlm, image_files)
            image_files = []
            for z in iterable:
                image_files.append(z)
            for m in range(shape[0]):
                arr[m] = image_files[m]
            io.volwrite(f'/dls/i13/data/2023/mg33497-1/processing/8bit_tiffs_cropped_nlm/{str(os.path.basename(Path(i)))}', arr.astype(np.uint8), format = 'tiff', bigtiff=True)

if __name__ == '__main__':
    main()
