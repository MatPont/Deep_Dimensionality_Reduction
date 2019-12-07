import os
from time import time

from sklearn import manifold

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from scipy import io
import numpy as np

dataset_folder = "./FCPS"

n_neighbors = 10



def execute_algo(algo, algo_name, subplot_cpt, y, fig):    
    print(algo_name,"...")
    t0 = time()
    X_transformed = algo().fit_transform(X)
    t1 = time()
    print(algo_name, ": %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(2, 6, subplot_cpt)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y.ravel(), cmap=plt.cm.rainbow)
    plt.title("%s (%.2g sec)" % (algo_name, t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')    



for myfile in os.listdir(dataset_folder):
    print("#####################")
    print("#",myfile)
    print("#####################")
    mat = io.loadmat(dataset_folder+"/"+myfile)
    X = mat["fea"]
    y = mat["gnd"]
    print(X.shape)
    print(y.shape)
    
    subplot_cpt = 1
    
    #####################
    # Plot dataset
    #####################
    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (X.shape[0], n_neighbors), fontsize=14)
    if X.shape[1] == 3:
        ax = fig.add_subplot(2, 6, subplot_cpt, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y.ravel(), cmap=plt.cm.rainbow)
        ax.view_init(40, -10)        
    elif X.shape[1] == 2:
        ax = fig.add_subplot(2, 6, subplot_cpt)
        ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.rainbow)
            
    
    #####################    
    # Multi-Dimensional Scaling 
    #####################    
    subplot_cpt += 2
    execute_algo(manifold.MDS, "Multi-Dimensional Scaling", subplot_cpt, y, fig)
    
    
    #####################    
    # ISOMAP
    #####################    
    subplot_cpt += 2  
    execute_algo(manifold.Isomap, "Isomap", subplot_cpt, y, fig)    
    
    
    #####################    
    # Locally Linear Embedding
    #####################    
    subplot_cpt += 4
    execute_algo(manifold.LocallyLinearEmbedding, "Locally Linear Embedding", subplot_cpt, y, fig)    
    
    
    #####################    
    # Laplacian Eigenmap
    #####################    
    subplot_cpt += 2 
    execute_algo(manifold.SpectralEmbedding, "Laplacian Eigenmap", subplot_cpt, y, fig)
        
    
    # Final plot
    #plt.show()
    #plt.savefig(myfile[:-4]+".png", format="png")
    plt.savefig(myfile[:-4]+".svg", format="svg")
