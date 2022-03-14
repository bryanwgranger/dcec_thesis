from time import time
import os
import numpy as np
import io
import boto3
import tensorflow as tf
import keras.backend as K
from keras import layers, losses
from keras.models import Model, Sequential

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
# from autoencoder import Autoencoder
# from cae2 import Autoencoder

def Autoencoder():

    model = Sequential([
        layers.InputLayer((128,128,3)),
        layers.Conv2D(32, (5, 5), activation='elu', padding='same', strides=2, name='conv1'),
        layers.Conv2D(64, (5, 5), activation='elu', padding='same', strides=2, name='conv2'),
        layers.Conv2D(128, (3, 3), activation='elu', padding='same', strides=2, name='conv3'),
        layers.Flatten(),
        layers.Dense(32, activation='elu', name='embedding'),
        # layers.InputLayer((32,)),
        layers.Dense(32768, activation='elu'),
        layers.Reshape((16,16,128)),
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='elu', padding='same', name='deconv1'),
        layers.Conv2DTranspose(64, kernel_size=5, strides=2, activation='elu', padding='same', name='deconv2'),
        layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation='elu', padding='same', name='deconv3'),
        layers.Conv2D(3, (3, 3), activation='elu', padding='same', name='reconstruction'),
    ])

    return model

class ClusteringLayer(layers.Layer):

    """
     Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
     sample belonging to each cluster. The probability is calculated with student's t-distribution.
     # Example
     ```
         model.add(ClusteringLayer(n_clusters=10))
     ```
     # Arguments
         n_clusters: number of clusters.
         weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
         alpha: parameter in Student's t-distribution. Default to 1.0.
     # Input shape
         2D tensor with shape: `(n_samples, n_features)`.
     # Output shape
         2D tensor with shape: `(n_samples, n_clusters)`.
     """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, 32))
        self.clusters = self.add_weight(shape=(self.n_clusters, 32), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class DCEC(object):
    def __init__(self,
                 input_shape,
                 # filters=[32, 64, 128, 10],
                 n_clusters=9,
                 alpha=1.0):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = Autoencoder()
        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(n_clusters=self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])


    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size * 5)
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(accuracy_score(y, self.y_pred), 5)
                    nmi = np.round(normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            ite += 1
            print("ite:", ite)

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)



if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--n_clusters', default=9, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/temp')
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--key', default='dataset.npz',type=str)
    parser.add_argument('--bucket', default='tf-numpy-dataset', type=str)
    args = parser.parse_args()

    tol = args.tol
    maxiter = args.maxiter
    update_interval = args.update_interval
    save_dir = args.save_dir
    cae_weights = args.cae_weights
    n_clusters = args.n_clusters
    batch_size = args.batch_size
    gamma = args.gamma
    sm_model_dir = args.sm_model_dir
    model_dir = args.model_dir
    training_dir = args.train
    key = args.key
    bucket = args.bucket

    ##read data
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, key)
    with io.BytesIO(obj.get()["Body"].read()) as f:
        f.seek(0)  # rewind the file
        dataset = np.load(f)

    ##build model
    dcec = DCEC(input_shape=(None,128,128,3), n_clusters=n_clusters)
    
    optimizer = 'adam'
    gamma = 0.1

    dcec.compile(loss=['kld', 'mse'], loss_weights=[gamma, 1], optimizer=optimizer)

    dcec.fit(x=dataset, y=None, tol=tol, maxiter=maxiter,
             update_interval=update_interval,
             save_dir=sm_model_dir,
             cae_weights=cae_weights)


