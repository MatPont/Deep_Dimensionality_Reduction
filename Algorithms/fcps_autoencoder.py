import sys
from tensorflow import keras
from tensorflow.keras import Input, layers, optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from scipy import io
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


class Autoencoder:
    def __init__(self, hidden_dim=128, architecture=2):
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'        
        
    def make_autoencoder_model(self, d):
    
        #################################################
        # Hyper-parameters
        #################################################
        encoding_dim = 2
        hidden_dim = self.hidden_dim
        architecture = self.architecture

        #activation='elu'
        activation='leaky_relu'
        optimizer=self.optimizer
        loss=self.loss        
        #optimizer=optimizers.Adam(lr=0.01)
        #loss='mean_squared_error'
        #################################################

        out_activation = 'sigmoid' if loss=='binary_crossentropy' else 'linear'
        activation = None if activation=='leaky_relu' else activation

        #################################################
        # Auto-encoder
        #################################################
        input_img = Input(shape=(d,))


        ####### Encoder #######
        encoded = input_img        
        #   Layer
        if architecture >= 1:
            encoded = Dense(hidden_dim, activation=activation)(encoded)
            if activation is None: encoded = layers.LeakyReLU()(encoded)

        #   Layer
        if architecture >= 2:        
            encoded = Dense(hidden_dim//2, activation=activation)(encoded)
            if activation is None: encoded = layers.LeakyReLU()(encoded)
            
        #   Layer
        if architecture >= 3:
            encoded = Dense(hidden_dim//4, activation=activation)(encoded)
            if activation is None: encoded = layers.LeakyReLU()(encoded)                    

        #   Layer
        encoded = Dense(encoding_dim, activation=out_activation)(encoded)


        ####### Decoder #######
        decoded = encoded
        #   Layer
        if architecture >= 3:        
            decoded = Dense(hidden_dim//4, activation=activation)(decoded)
            if activation is None: decoded = layers.LeakyReLU()(decoded)   
                    
        #   Layer
        if architecture >= 2:        
            decoded = Dense(hidden_dim//2, activation=activation)(decoded)
            if activation is None: decoded = layers.LeakyReLU()(decoded)

        #   Layer
        if architecture >= 1:
            decoded = Dense(hidden_dim, activation=activation)(decoded)
            if activation is None: decoded = layers.LeakyReLU()(decoded)         

        #   Layer
        decoded = Dense(d, activation=out_activation)(decoded)
        
        
        ####### Make model #######
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        #decoder = Model(encoded, decoded)
        
        autoencoder.compile(optimizer=optimizer, loss=loss)

        return autoencoder, encoder, decoded


    def fit_transform(self, X):
    
        #################################################
        # Hyper-parameters
        #################################################
        epoch = 300
        batch_size = 64   
        
        optimizer=self.optimizer
        loss=self.loss         
        #################################################        
    
        X = (X - X.min(0))/(X.max(0) - X.min(0))
        d = X.shape[1]

        autoencoder, encoder, decoded = self.make_autoencoder_model(d)

        ####### Plot model #######
        autoencoder.summary()


        ####### Train #######
        autoencoder.fit(X, X,
                        epochs=epoch,
                        batch_size=batch_size,
                        shuffle=True, verbose=0)
        
        return encoder.predict(X)
        
        
if __name__ == "__main__":
    mat = io.loadmat(sys.argv[1])
    X = mat["fea"]
    y = mat["gnd"]

    print(sys.argv[1])
    print(X.shape)
    
    if len(sys.argv) <= 2:
        param_hidden_dim = [16,32,64]
        param_architecture = [0,1,2]
    else:
        param_hidden_dim = [int(sys.argv[2])]
        param_architecture = [int(sys.argv[3])]        
        
    fig = plt.figure()    
    subplot_cpt=1
    subplot_row=len(param_architecture)
    subplot_col=len(param_hidden_dim)    
    
    for architecture in param_architecture:
        for hidden_dim in param_hidden_dim:
            print("Architecture = ",architecture)
            print("Hidden dim = ",hidden_dim)            
            X_transformed = Autoencoder(hidden_dim=hidden_dim,
                                        architecture=architecture).fit_transform(X)   
  
            ax = fig.add_subplot(subplot_row, subplot_col, subplot_cpt)
            subplot_cpt += 1            
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y.ravel(), cmap=plt.cm.rainbow)
            plt.title("A=%i - HD=%i" % (architecture, hidden_dim))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())            
            plt.axis('tight')      
            
            if architecture==0: break; 

    plt.show()
    
