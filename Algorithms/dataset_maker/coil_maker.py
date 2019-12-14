import sys
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from scipy import io, sparse

def get_sample(path):
    data = []
    all_files = sorted(os.listdir(path))
    for file in all_files:
        if file[-4:] == ".png":
            print(file)        
            data.append(rgb2gray(plt.imread(path + '/' + file)))
    data = np.array(data)
    images = data
    data = data.reshape(data.shape[0], 128*128)
    return data, images

path = sys.argv[1]
sample_data, sample_images = get_sample(path)
sample_data.shape
sample_images.shape

csr_mat = sparse.csr_matrix(sample_data)
csr_mat.eliminate_zeros()

io.savemat(sys.argv[2]+".mat", {'X': csr_mat})
