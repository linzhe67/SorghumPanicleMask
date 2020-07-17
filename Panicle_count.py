import os
import functools
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)
import cv2
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import models
from tensorflow.python.keras.utils import CustomObjectScope
from tensorflow.python.keras.initializers import glorot_uniform

def _test_process_pathnames(fname):
  # We map this function onto each pathname pair  
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    return img

def test_shift_img(output_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([], 
                                              -width_shift_range * test_img_shape[1],
                                              width_shift_range * test_img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                               -height_shift_range * test_img_shape[0],
                                               height_shift_range * test_img_shape[0])
        # Translate both 
        output_img = tfcontrib.image.translate(output_img,
                                             [width_shift_range, height_shift_range])
        
    return output_img

def test_flip_img(horizontal_flip, tr_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img)),
                                lambda: (tr_img))
    return tr_img

def _test_augment(
             test_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             #horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically 
    if resize is not None:
        test_img = tf.image.resize_images(test_img, resize)
  
    if hue_delta:
        test_img = tf.image.random_hue(test_img, hue_delta)
  
    #test_img = test_flip_img(horizontal_flip, test_img)
    test_img = test_shift_img(test_img, width_shift_range, height_shift_range)
    test_img = tf.to_float(test_img) * scale
    return test_img

def test_get_baseline_dataset(filenames, 
                         preproc_fn=functools.partial(_test_augment),
                         threads=5, 
                         batch_size=1,
                         shuffle=False):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_test_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)
  
  
    # It's necessary to repeat our data for all epochs 
    dataset = dataset.repeat().batch(batch_size)
    return dataset

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    
    model_path = 'model_name.hdf5'

    
    with tf.device('/cpu:0'):
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = models.load_model(model_path)
    
    home = 'testing_directory'
    test_dir = os.path.join(home, "testing_dataset")
    test_file = os.listdir(test_dir)
    test_files = []
    for img in range(len(test_file)):
        test_files.append(os.path.join(test_dir, test_file[img]))
        

    test_img_shape = (1024, 1024, 3)
    batch_size = 1
    
    test_cfg = {
        'resize': [test_img_shape[0], test_img_shape[1]],
        'scale': 1 / 1023.,
        'hue_delta': 0.1
    }
    test_preprocessing_fn = functools.partial(_test_augment, **test_cfg)
    
    test_ds = test_get_baseline_dataset(test_files, 
                                  preproc_fn=test_preprocessing_fn,
                                  batch_size=1)
    
    test_data_aug_iter = test_ds.make_one_shot_iterator()
    test_next_element = test_data_aug_iter.get_next()
    
    
    plt.figure(figsize=(7.12, 7.12)) 
    for i in range(len(test_files)):
        batch_of_imgs= tf.keras.backend.get_session().run(test_next_element)
        img = batch_of_imgs[0]
        predicted_label = model.predict(batch_of_imgs)[0]
        
        plt.imshow(predicted_label[:, :, 0])
        plt.axis('off')
        plt.savefig("multi500_test_{}_{}.png".format(i,i),bbox_inches = 'tight',pad_inches = 0)
    #plt.show()

    n = 0 
    plt.figure(figsize=(7.12, 7.12))
    for i in range(len(test_files)):
        img = cv2.imread("multi500_test_{}_{}.png".format(i,i),cv2.IMREAD_UNCHANGED)

        ret,thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)
        
        image, contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
         
        for c in contours:
            if cv2.contourArea(c) >= 200 and cv2.contourArea(c) <= 5000:

                img3=cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        n = n + len(contours)
        
    print("The Total Sorghum heads number are:", n)


