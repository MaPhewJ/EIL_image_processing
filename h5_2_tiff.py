import h5py
import imageio as io
import numpy as np

#Often H5 files are nonobvious to work with if you dont know how they are structured
#Can be converted to normal image files (.tiff) using something like below 

# Open the H5 file
myfile = h5py.File('path/to/myfile.h5', 'r')

# Keys
list(myfile.keys())

# Access the dataset containing the image data
dataset = myfile['dataset_name']  # Replace 'dataset_name' with the actual dataset name output by the line above
#OR
#dataset = myfile['dataset_name']['data_subset_name'] # if h5 file may have multiple layers

# Read the image data into a NumPy array
image_data = dataset[:]

# Convert the data type if necessary (optional)
image_data = image_data.astype(np.uint16)  # Example conversion to uint16

# Save the 3D TIFF image
io.volwrite('output.tiff', image_data, format = 'tiff')