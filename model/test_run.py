from DCEC import DCEC
import numpy as np
from keras.utils.vis_utils import plot_model

# load dataset
with open('/Users/bryanwgranger/Desktop/bryan/DS/thesis/code/dataset/np/dataset.npz', 'rb') as f:
    dataset = np.load(f)

print(dataset.shape)

def main():
    dcec = DCEC(input_shape=(None,128,128,3), n_clusters=5)
    plot_model(dcec.model, to_file='dcec_model.png', show_shapes=True)
    print(dcec.model.summary())

    optimizer = 'adam'
    gamma = 0.1

    dcec.compile(loss=['kld', 'mse'], loss_weights=[gamma, 1], optimizer=optimizer)

    ##args
    tol = 0.001
    maxiter = 2e4
    update_interval = 140
    save_dir = 'save'
    cae_weights = None

    dcec.fit(x=dataset, y=None, tol=tol, maxiter=maxiter,
             update_interval=update_interval,
             save_dir=save_dir,
             cae_weights=cae_weights)


    print('done')

main()