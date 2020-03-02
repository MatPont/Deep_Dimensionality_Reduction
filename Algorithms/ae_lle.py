from utils import generate_and_save_images

import tensorflow as tf
from scipy import sparse
from time import time


mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


class AE_LLE():
    def __init__(self, autoencoder, encoder, LLE, batch_size=128, mode="full", 
                 l_reg=1.00, verbose=False, train_both=True):
        """mode : {"batch", "full"} 
                  be careful, LLE in batch mode will not take the full dataset 
                  in input hence its global structure
           train_both: whether minimize both terms of loss at the same time or
                       one at a time (better results with True)"""
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.lle = LLE
        self.batch_size = batch_size
        self.ae_lle_optimizer = self.make_ae_lle_optimizer()
        self.mode = mode
        self.l_reg = l_reg
        self.verbose = verbose
        self.train_both = train_both

    def get_X_encoded(self, X):
        return self.encoder.predict(X)

    def get_Y_lle(self):
        return self.lle.get_Y()

    def encoder_loss(self, X_encoded, W, use_sparse=False):
        if use_sparse:
            X_encoded_sparse = sparse.csr_matrix(X_encoded)
            WX = W.dot(X_encoded_sparse).todense()
        else:
            #WX = W.dot(X_encoded)            
            WX = tf.matmul(tf.convert_to_tensor(W, dtype=tf.float32), X_encoded)
        return self.l_reg * mse(X_encoded, WX)

    def ae_lle_loss(self, X, X_decoded, X_encoded, W):
        return cross_entropy(X, X_decoded) + self.l_reg * self.encoder_loss(X_encoded, W)

    def make_ae_lle_optimizer(self):
        return tf.keras.optimizers.Adam()
        #return tf.keras.optimizers.Adam(2e-3)

    # Train encoder according to the second term of the loss
    # Or both autoencoder and encoder if train_both=True
    @tf.function
    def ae_lle_train_step(self, X, W, use_sparse=False, train_both=False):
        with tf.GradientTape() as ae_lle_tape:
          X_encoded = self.encoder(X, training=True)
          if not train_both:
              loss = self.encoder_loss(X_encoded, W, use_sparse=use_sparse)
          else:
              X_decoded = self.autoencoder(X, training=True)
              loss = self.ae_lle_loss(X, X_decoded, X_encoded, W)

        if not train_both:
            gradients_of_ae_lle = ae_lle_tape.gradient(loss, self.encoder.trainable_variables)
            self.ae_lle_optimizer.apply_gradients(zip(gradients_of_ae_lle, self.encoder.trainable_variables))
        else:
            gradients_of_ae_lle = ae_lle_tape.gradient(loss, self.autoencoder.trainable_variables)
            self.ae_lle_optimizer.apply_gradients(zip(gradients_of_ae_lle, self.autoencoder.trainable_variables))          

        return loss

    def fit(self, X, epoch=50, ae_step=1):
        t0 = time()
        if self.mode == "batch":
            epoch = epoch * batch_size
        ####### Train #######
        for ep in range(epoch):
            X_batch = X
            if self.mode == "batch":
                indexes = np.random.randint(low=0,high=X.shape[0],size=batch_size)
                X_batch = X[indexes]

            for i in range(ae_step):   
                # Autoencoder Training
                if not self.train_both:
                    if self.verbose: print("--- Autoencoder update...")
                    tt0 = time()
                    history = self.autoencoder.fit(X_batch, X_batch,
                                    epochs=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, verbose=0)
                    results = history.history['loss'][0]
                    #results = self.autoencoder.evaluate(X, X, batch_size=self.batch_size, verbose=0)    
                    tt1 = time()
                    if self.verbose: print("(%.2g sec)"%(tt1 - tt0))

                # LLE Training
                if i == 0: # fit only on the first ae_step
                    if self.verbose: print("encode data...")
                    X_encoded = self.encoder.predict(X_batch)
                    if self.verbose: print(X_encoded.shape)

                    if self.verbose: print("--- LLE update...")
                    tt0 = time()
                    self.lle.fit(X_encoded)
                    W = self.lle.get_W()
                    tt1 = time()
                    if self.verbose: print("(%.2g sec)"%(tt1 - tt0))

                # Autoencoder/Encoder Training
                if self.verbose: print("--- Autoencoder/Encoder update...")
                tt0 = time()
                encoder_loss = self.ae_lle_train_step(X_batch, W.todense(), train_both=self.train_both)
                #encoder_loss = self.ae_lle_train_step(X_batch, W.todense())
                #encoder_loss = self.ae_lle_train_step(X_batch, W, use_sparse=True)
                tt1 = time()
                if self.verbose: print("(%.2g sec)"%(tt1 - tt0))              

            if ep % (epoch//20) == 0:
                t1 = time()
                print("_________________________________________________________________")
                print("epoch : ",ep, " / ", epoch, "(%.2g sec)"%(t1 - t0))
                if self.train_both:
                  print("Autoencoder loss = ", encoder_loss.numpy())                
                else:
                  print("Autoencoder loss = ", results)             
                  print("Encoder loss     = ", encoder_loss.numpy())                
                print("LLE loss         = ", self.lle.compute_weight_loss(X_encoded))            

        """for _ in range(8):
          generate_and_save_images(self.autoencoder, 0, 
                                test_input=X[np.random.randint(low=0,high=X.shape[0],size=16)])"""

        X_encoded = self.encoder.predict(X)
        self.lle.fit(X_encoded)

        if self.train_both:
          print("Autoencoder loss = ", encoder_loss.numpy())                
        else:
          print("Autoencoder loss = ", results)             
          print("Encoder loss     = ", encoder_loss.numpy())   
        print("LLE loss         = ", self.lle.compute_weight_loss(X_encoded))            

    def transform(self, X):
        pass     

    def fit_transform(self, X):
        pass           



"""
encoder_losses = []
# Batch Encoder Training
for it in range(int(X_train.shape[0] / batch_size)):
    indexes = np.random.randint(low=0,high=X_train.shape[0],size=batch_size)
    X_batch = X_train[indexes]  
    W_batch = W[indexes]
    print(W_batch.shape)
    print(X_batch.shape)
    input()
    encoder_loss = ae_lle_train_step(X_batch, W_batch)
    encoder_losses.append(encoder_loss)
print("Encoder loss = ", mean(encoder_losses))"""
