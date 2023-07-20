import numpy as np
import imageio as io 
from glob import glob
from scipy.ndimage import rotate
from tqdm import tqdm

#When making ML models from small datasets it can be usefull to augment the dataset
#This basically make it artificially larger by rotating, flipping or other actions...

og_img = glob('D:/fuelcell/train_images/*.tif')
rotated = []
flipped = []
rot_flip = []

for i in tqdm(og_img):
    im = io.imread(i)
    rotated.append(rotate(im, 180))
    flipped.append(np.fliplr(im))
    rot_flip.append(rotate(np.fliplr(im),180))

for i,p in tqdm(zip(rotated,og_img)):
    io.imwrite(p[:-4]+"rotated.tiff",i,format='tiff')

for i,p in tqdm(zip(flipped,og_img)):
    io.imwrite(p[:-4]+"flipped.tiff",i,format='tiff')

for i,p in tqdm(zip(rot_flip,og_img)):
    io.imwrite(p[:-4]+"rot_flip.tiff",i,format='tiff')


og_img = glob('D:/fuelcell/train_labels/*.tiff')
rotated = []
flipped = []
rot_flip = []

for i in tqdm(og_img):
    im = io.imread(i)
    rotated.append(rotate(im, 180))
    flipped.append(np.fliplr(im))
    rot_flip.append(rotate(np.fliplr(im),180))

for i,p in tqdm(zip(rotated,og_img)):
    io.imwrite(p[:-5]+"rotated.tiff",i,format='tiff')

for i,p in tqdm(zip(flipped,og_img)):
    io.imwrite(p[:-5]+"flipped.tiff",i,format='tiff')

for i,p in tqdm(zip(rot_flip,og_img)):
    io.imwrite(p[:-5]+"rot_flip.tiff",i,format='tiff')



