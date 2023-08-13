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
import h5py
from numpy import sqrt

#Application of parralelised non-local means filter to images of electrode particles during mg35714-1 beamtime. 
#This will check if the image has already been processed before running and can therefore be run continuously 

def apply_nlm(image):
    img = img_as_float(image) ###load image to be smoothed by NLM.
    img = denoise_nl_means(img, h=0.00472*2.5, fast_mode=True, patch_size=10, patch_distance=15, channel_axis=None)
    return (img*255).astype(np.uint8)
    
def main():
    processed = []
    for i in glob('/dls/i13/data/2023/mg35714-1/processing/batch_processed_matthew/*.tiff'):
        processed.append(str(os.path.basename(Path(i)))[:6])
    for i in tqdm(glob('/dls/i13/data/2023/mg35714-1/processing/reconstruction/*/*/*tomopy_recon_r.h5')):
        if str(Path(i))[77:83] in processed:
            pass
        elif int(str(Path(i))[77:83]) < 163500: #lower range scan number optional
            pass
        elif int(str(Path(i))[77:83]) > 163550: #upper range scan number optional
            pass 
        else:
            print(f'processing {i}')
            myfile = h5py.File(i, 'r')
            vol = myfile['4-TomopyRecon-tomo']['data'] #load data from h5
            vol = vol[:]
            amiin = sqrt(np.amin(vol)**2)
            mean = np.mean((vol+amiin))
            std = np.std(vol+amiin)
            maax = mean + 5*std
            miin = mean - 5*std
            resliced = []
            for k in range(vol.shape[1]): #reslice so that vertically sliced 
                im = ((vol[:,k,:]+amiin)/(maax)) #convert to 8bit
                im = np.where(im < (miin/maax), 0, im)
                im = np.where(im > 1, 1, im)*255
                im = im - ((mean/maax)*225 - 127.5)
                resliced.append((im).astype(np.uint8))
            resliced = np.array(resliced)[:,:,750:1750] #cropping
            shape = resliced.shape #prepping data for parralel processing
            arr = np.empty(shape, np.uint8)
            shape = list(resliced.shape)
            image_files = []
            for l in range(shape[0]):###loops through tomo slices create image list conc futures
                image_files.append(resliced[l])
            with concurrent.futures.ProcessPoolExecutor() as executor:
                iterable = executor.map(apply_nlm, image_files)
            image_files = []
            for z in iterable:
                image_files.append(z)
            for m in range(shape[0]):
                arr[m] = image_files[m]
            io.volwrite(f'/dls/i13/data/2023/mg35714-1/processing/batch_processed_matthew/{str(Path(i))[77:83]}.tiff', arr.astype(np.uint8), format = 'tiff', bigtiff=True) #saving

if __name__ == '__main__':
    main()
