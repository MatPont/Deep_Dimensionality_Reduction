import tensorflow as tf
from scipy import sparse

from utils import generate_and_save_images


mse = tf.keras.losses.MeanSquaredError()


class AE_LLE():
    def __init__(self, autoencoder, encoder, LLE, batch_size=256):
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.lle = LLE
        self.batch_size = batch_size
        self.ae_lle_optimizer = self.make_ae_lle_optimizer()

    def ae_lle_loss(self, X_encoded, W):
        X_encoded_sparse = sparse.csr_matrix(X_encoded)
        return mse(X_encoded, W.dot(X_encoded_sparse).todense())

    def make_ae_lle_optimizer(self):
        return tf.keras.optimizers.Adam()

    # Train encoder according to the second term of the loss
    def ae_lle_train_step(self, X, W):
        with tf.GradientTape() as ae_lle_tape:
          X_encoded = self.encoder(X, training=True)
          encoder_loss = self.ae_lle_loss(X_encoded, W)

        gradients_of_ae_lle = ae_lle_tape.gradient(encoder_loss, self.encoder.trainable_variables)

        self.ae_lle_optimizer.apply_gradients(zip(gradients_of_ae_lle, self.encoder.trainable_variables))

        return encoder_loss

    def fit(self, X, epoch=50):
        ####### Train #######
        for ep in range(epoch):
            print("_________________________________________________________________")
            print("epoch : ",ep, " / ", epoch)
            #losses = {"Autoencoder":[], "LLE":[], "AE-LLE":[]}

            # Autoencoder Training
            #print("--- Autoencoder update...")
            self.autoencoder.fit(X, X,
                            epochs=1,
                            batch_size=batch_size,
                            shuffle=True, verbose=0)
            results = self.autoencoder.evaluate(X, X, batch_size=self.batch_size, verbose=0)
            print("Autoencoder loss = ", results)
            
            # LLE Training
            #print("--- LLE update...")
            X_encoded = self.encoder.predict(X_train)
            #print(X_encoded.shape)
            self.lle.fit(X_encoded)
            print("LLE loss         = ", self.lle.compute_weight_loss(X_encoded))

            # Encoder Training
            #print("--- Encoder update...")
            encoder_losses = []
            W = self.lle.get_W()
            encoder_loss = self.ae_lle_train_step(X_train, W)
            print("Encoder loss     = ", encoder_loss.numpy())

            """# Batch Encoder Training
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

        for _ in range(8):
          generate_and_save_images(self.autoencoder, 0, 
                                test_input=X[np.random.randint(low=0,high=X.shape[0],size=16)])

    def transform(self, X):
        pass     

    def fit_transform(self, X):
        pass  
