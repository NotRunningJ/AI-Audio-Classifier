from process_data import *              
from autoencoder import *         # current around 100%!!! with 20 folders of data and 10 epochs
from cnn import *


# import warnings for reading in data
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# read the data in
train_ds, train_labels, train_ds_noisy, train_labels_noisy, test_ds, test_labels, test_ds_noisy, \
        test_labels_noisy, val_ds, val_labels, val_ds_noisy, val_labels_noisy = preprocess_data(1)

mfcc_shape = train_ds[0].shape

# train and denoise data with autoencoder
autoencoder = train_autoencoder(train_ds_noisy, train_ds, val_ds_noisy, val_ds)
train_decoded = denoise_data(autoencoder, train_ds_noisy)
test_decoded = denoise_data(autoencoder, test_ds_noisy)
val_decoded = denoise_data(autoencoder, val_ds_noisy)

# train and classify denoised data
model = create_cnn(train_decoded[0].shape)
model = train_model(model, train_decoded, train_labels, val_decoded, val_labels)
test_model(model, test_decoded, test_labels)