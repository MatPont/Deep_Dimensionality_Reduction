import sys
import os
from time import time

from sklearn import manifold, decomposition
from fcps_autoencoder import Autoencoder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from scipy import io
import numpy as np


dataset_folder = "../Datasets/FCPS"

subplot_row = 2
subplot_col = 8


def add_subplot(algo_name, subplot_cpt, y, fig, t0, t1, X_transformed):
    ax = fig.add_subplot(subplot_row, subplot_col, subplot_cpt)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y.ravel(), cmap=plt.cm.rainbow)
    plt.title("%s (%.2g sec)" % (algo_name, t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')    

def execute_algo(algo, X, algo_name, subplot_cpt, y, fig):    
    print(algo_name,"...")
    t0 = time()
    X_transformed = algo().fit_transform(X)
    t1 = time()
    print(algo_name, ": %.2g sec" % (t1 - t0))
    add_subplot(algo_name, subplot_cpt, y, fig, t0, t1, X_transformed)
    return X_transformed


if __name__ == "__main__":

    mode = len(sys.argv)

    for myfile in os.listdir(dataset_folder):
        if mode >= 2 and myfile != sys.argv[1]+".mat": continue
        
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
        fig = plt.figure(figsize=(15, 5))
        """plt.suptitle(myfile[:-4]+" Dataset (%i points and %i classes)"
                     % (X.shape[0], len(np.unique(y))), fontsize=14)"""
        if X.shape[1] == 3:
            ax = fig.add_subplot(subplot_row, subplot_col, subplot_cpt, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y.ravel(), cmap=plt.cm.rainbow)
            ax.view_init(40, -10)        
        elif X.shape[1] == 2:
            ax = fig.add_subplot(subplot_row, subplot_col, subplot_cpt)
            ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.rainbow)


        #####################    
        # PCA
        #####################    
        subplot_cpt += 2
        execute_algo(decomposition.PCA, X, "PCA", subplot_cpt, y, fig)  
                    
        
        #####################    
        # Multi-Dimensional Scaling 
        #####################    
        subplot_cpt += 2
        execute_algo(manifold.MDS, X, "Multi-Dimensional Scaling", subplot_cpt, y, fig)
        
        
        #####################    
        # Locally Linear Embedding
        #####################    
        subplot_cpt += 2
        execute_algo(manifold.LocallyLinearEmbedding, X, "Locally Linear Embedding", subplot_cpt, y, fig)
        
        
        #####################        
        # New line in plot
        #####################        
        subplot_cpt += 2
        

        #####################    
        # Deep Autoencoder
        #####################         
        subplot_cpt += 2 
        execute_algo(Autoencoder, X, "Autoencoder", subplot_cpt, y, fig)

        
        #####################    
        # Laplacian Eigenmap
        #####################    
        subplot_cpt += 2 
        execute_algo(manifold.SpectralEmbedding, X, "Laplacian Eigenmap", subplot_cpt, y, fig)


        #####################    
        # ISOMAP
        #####################    
        subplot_cpt += 2  
        execute_algo(manifold.Isomap, X, "Isomap", subplot_cpt, y, fig)    
            
        # Final plot
        #plt.show()
        plt.subplots_adjust(left=0.05, right=1.05)
        plt.savefig(myfile[:-4]+".png", format="png")
        plt.savefig(myfile[:-4]+".svg", format="svg")
        
        if mode >= 2: break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""size=100
alpha=1
for myfile in os.listdir(dataset_folder):
    if mode >= 2 and myfile != sys.argv[1]+".mat": continue
    
    print("#####################")
    print("#",myfile)
    print("#####################")
    mat = io.loadmat(dataset_folder+"/"+myfile)
    X = mat["fea"]
    y = mat["gnd"]
    print(X.shape)
    print(y.shape)
    
    #####################
    # Plot dataset
    #####################
    fig = plt.figure(figsize=(10, 10))
    #plt.suptitle(myfile[:-4]+" Dataset (%i points and %i classes)"
    #             % (X.shape[0], len(np.unique(y))), fontsize=14)
    if X.shape[1] == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y.ravel(), cmap=plt.cm.rainbow, s=size, alpha=alpha)
        ax.view_init(40, -10)        
        plt.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)                
    elif X.shape[1] == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.rainbow, s=size, alpha=alpha)

    plt.savefig(myfile[:-4]+"2.png", format="png")
    plt.savefig(myfile[:-4]+"2.svg", format="svg") 
    
    
exit()"""
