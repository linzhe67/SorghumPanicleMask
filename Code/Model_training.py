# -*- coding: utf-8 -*-
import os
import functools

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from sklearn.model_selection import train_test_split

img_shape = (1024, 1024, 3)
batch_size = 3
epochs = 15

home = 'directory'

img_dir = os.path.join(home, "training")
mask_dir = os.path.join(home, "training_masks")

train_file = os.listdir(img_dir)
mask_file = os.listdir(mask_dir)

train_files = []
mask_files = []
for img in range(len(train_file)):
    train_files.append(os.path.join(img_dir, train_file[img]))
    mask_files.append(os.path.join(mask_dir, mask_file[img]))
    
train_files, train_val_files, mask_files, mask_val_files = \
                    train_test_split(train_files, mask_files, test_size=0.1)

num_train_examples = len(train_files)
num_val_examples = len(mask_files)

def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair  
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    label_img_str = tf.read_file(label_path)
    # These are gif images so they return as (num_frames, h, w, c)
    label_img = tf.image.decode_gif(label_img_str)[0]
    # The label image should only have values of 1 or 0, indicating pixel wise
    # object (car) or not (background). We take the first channel only. 
    label_img = label_img[:, :, 0]
    label_img = tf.expand_dims(label_img, axis=-1)
    return img, label_img

def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([], 
                                              -width_shift_range * img_shape[1],
                                              width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                               -height_shift_range * img_shape[0],
                                               height_shift_range * img_shape[0])
        # Translate both 
        output_img = tfcontrib.image.translate(output_img,
                                             [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(label_img,
                                             [width_shift_range, height_shift_range])
    return output_img, label_img

def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
    return tr_img, label_img

def _augment(img,
             label_img,
             resize=None,  
             scale=1,  
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically 
    if resize is not None:
    # Resize both images
        label_img = tf.image.resize_images(label_img, resize)
        img = tf.image.resize_images(img, resize)
  
    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)
  
    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
    label_img = tf.to_float(label_img) * scale
    img = tf.to_float(img) * scale 
    return img, label_img

def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=batch_size,
                         shuffle=True):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)
  
  
    # It's necessary to repeat our data for all epochs 
    dataset = dataset.repeat().batch(batch_size)
    return dataset

tr_cfg = {
    'scale': 1 / 1023.,
    'hue_delta': 0.1,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {
    'scale': 1 / 1023.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

train_ds = get_baseline_dataset(train_files,
                                mask_files,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)
val_ds = get_baseline_dataset(train_files,
                              mask_files, 
                              preproc_fn=val_preprocessing_fn,
                              batch_size=batch_size)

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

inputs = layers.Input(shape=img_shape)
#1024

encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 512

encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 256

encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 128

encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 64

encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
# 32

encoder5_pool, encoder5 = encoder_block(encoder4_pool, 1024)
# 16

encoder6_pool, encoder6 = encoder_block(encoder5_pool, 2048)
# 8

center = conv_block(encoder6_pool, 4096)
# center

decoder6 = decoder_block(center, encoder6, 2048)
# 16

decoder5 = decoder_block(decoder6, encoder5, 1024)
# 32

decoder4 = decoder_block(decoder5, encoder4, 512)
# 64

decoder3 = decoder_block(decoder4, encoder3, 256)
# 128

decoder2 = decoder_block(decoder3, encoder2, 128)
# 256

decoder1 = decoder_block(decoder2, encoder1, 64)
# 512

decoder0 = decoder_block(decoder1, encoder0, 32)
# 1024

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)


model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


save_model_path = os.path.join(home,"model_name.hdf5")

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                   epochs=epochs,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(num_val_examples / float(batch_size))),
                   callbacks=[cp])

