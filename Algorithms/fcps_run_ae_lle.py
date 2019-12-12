from fcps_run import execute_algo
from fcps_autoencoder import Autoencoder
from lle import LLE
from ae_lle import AE_LLE

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy import io
import numpy as np


dataset_folder = "../Datasets/FCPS"

subplot_row = 1
subplot_col = 6


def run_kMeans(X, y, name):
    kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=20)
    kmeans.fit(X)
    print(name)
    print(nmi(kmeans.labels_, y))
    print(ari(kmeans.labels_, y))


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
        # Make figure
        #####################
        fig = plt.figure(figsize=(15, 5))
        
        
        #####################    
        # Locally Linear Embedding
        #####################    
        #subplot_cpt += 2
        X_lle = execute_algo(manifold.LocallyLinearEmbedding, X, "Locally Linear Embedding", subplot_cpt, y, fig)
        

        #####################    
        # Autoencoder
        #####################         
        subplot_cpt += 2 
        X_ae = execute_algo(Autoencoder, X, "Autoencoder", subplot_cpt, y, fig)


        #####################    
        # AE LLE
        #####################
        subplot_cpt += 2 
        autoencoder, encoder, decoded = Autoencoder().make_autoencoder_model(X.shape[1])
        lle = LLE(neighbors_update=True, verbose=False)
        ae_lle = AE_LLE(autoencoder, encoder, lle)
        #ae_lle.fit(X)
        #X_ae_lle = execute_algo(TODO, X, "AE LLE", subplot_cpt, y, fig)


        #####################    
        # k-Means
        #####################
        # Classical kMeans    
        run_kMeans(X, y, "kMeans")
        
        # kMeans on LLE
        run_kMeans(X_lle, y, "LLE")
        
        # kMeans on AE
        run_kMeans(X_ae, y, "AE")
        
        # kMeans on AE LLE
        run_kMeans(X_ae_lle, y, "AE LLE")
        
        
        #####################      
        # Final plot
        #####################      
        #plt.show()
        plt.subplots_adjust(left=0.05, right=1.05)
        plt.savefig(myfile[:-4]+".png", format="png")
        plt.savefig(myfile[:-4]+".svg", format="svg")
        
        if mode >= 2: break
