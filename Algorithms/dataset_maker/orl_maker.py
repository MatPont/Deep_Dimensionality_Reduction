import sys
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from scipy import io, sparse

def get_sample(path):
    data = []
    for mydir in sorted(os.listdir(path)):
        mydir = path+mydir
        if os.path.isdir(mydir):
            for myfile in sorted(os.listdir(mydir)):
                if myfile[-4:] == ".pgm":
                    print("--- ",myfile)
                    data.append(rgb2gray(plt.imread(mydir + '/' + myfile)))
    data = np.array(data)
    images = data
    data = data.reshape(data.shape[0], 112*92)
    return data, images

path = sys.argv[1]
sample_data, sample_images = get_sample(path)
sample_data.shape
sample_images.shape

csr_mat = sparse.csr_matrix(sample_data)
csr_mat.eliminate_zeros()

print(csr_mat.shape)

io.savemat(sys.argv[2]+".mat", {'X': csr_mat})
