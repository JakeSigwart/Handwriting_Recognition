#Helper Functions for Machine Learning. Functions to Load data from files and pre-process that data.
import os
import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

###############################################################################################################################
#Input: A 1-D array of integer class labels, the number of classes
#Output: A 2-D array of shape: [len(class_labels), num_classes]
def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1
    return np.eye(num_classes, dtype=float)[class_numbers]

#Retreiving Data#################################################################################################################
#Re-constructs the object hierarchy from a file
def _unpickle_data(file_name):
    with open(file_name, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

#Pickle up to 4 variables and store at the given file-path
def _pickle_data(file_path, var1, var2=None, var3=None, var4=None):
    output = open(file_path, 'wb')
    pickle.dump(var1, output)
    pickle.dump(var2, output)
    pickle.dump(var3, output)
    pickle.dump(var4, output)
    output.close()

#Load a numpy array from a .npy file
def load_numpy_array(file_path):
    output = np.load(file_path)
    return output

#Save a numpy array in a .npy file
def save_numpy_array(my_array, file_path):
    np.save(file_path, my_array)
    
#Load a PNG image and return as a numpy array
def load_png_as_numpy(file_path):
    img = Image.open(file_path)
    arr = np.array(img)
    return arr

#More general function for extracting batches of cifar or MNIST data with the option to return one-hot labels
#Input: data-set, integer labels, [batch_size, height, width, num_channels], number of classes (only necessary for one-hot), one-hot-output
#Output: the data batch, labels (optionally one-hot)
def extract_random_image_batch(data, labels, data_batch_shape, one_hot=False, num_classes=0):
    batch_size = data_batch_shape[0]
    height = data_batch_shape[1]
    width = data_batch_shape[2]
    num_channels = data_batch_shape[3]
    
    random_indexes = np.zeros(shape=[batch_size], dtype=int)
    lbls = np.zeros(shape = [batch_size], dtype=int)
    batch = np.zeros(shape= [batch_size, height, width, num_channels], dtype=float)
    for n in range(batch_size):
        random_indexes[n] = np.random.randint(0, (len(data)-1))
        batch[n,:,:,:] = data[random_indexes[n],:,:,:]
        lbls[n] = labels[random_indexes[n]]
    if one_hot:
        lbls = one_hot_encoded(lbls, num_classes)
    return batch, lbls

#Cifar-10##########################################################################################################################
#Input: Training data path without number
#Get class names, cifar10 images, one-hot labels
def load_cifar10_data(file_path):
    raw = _unpickle_data(file_path + "\\batches.meta")[b'label_names']
    class_names = [x.decode('utf-8') for x in raw]
    print("Cifar10 class names loaded.")
    images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    cls = np.zeros(shape=[50000], dtype=int)
    begin = 0
    for i in range(5):
        data = _unpickle_data(file_path + "\\data_batch_" + str(i + 1))
        raw_image_batch = data[b'data']
        cls_batch = np.array(data[b'labels'])
        raw_image_batch = np.array(raw_image_batch, dtype=float) / 255.0
        raw_image_batch = raw_image_batch.reshape([-1, 3, 32, 32])
        raw_image_batch = raw_image_batch.transpose([0, 2, 3, 1])
        end = begin + len(raw_image_batch)
        images[begin:end,:,:,:] = raw_image_batch
        cls[begin:end] = cls_batch
        begin = end
        print("data_batch_" + str(i+1) + " Loaded")
    return class_names, images, cls


#Cifar-100###############################################################################################################################
#Load the fine class names 
def load_cifar100_class_names(file_path):
    raw = _unpickle_data(file_path)[b'fine_label_names']
    class_names = [x.decode('utf-8') for x in raw]
    return class_names

#Input: The complete name of the filepath (there is only 1 training and 1 test file)
def load_cifar100_data(filename):
    batchdata = _unpickle_data(filename)
    labels = np.array(batchdata[b'fine_labels'])
    images = batchdata[b'data']
    images = np.array(images, dtype=float) / 255.0
    images = images.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images, labels

#Input: An array of dimensions: [height, width, num_channels]
#Output: Display an RGB image
def plot_image(image):
    plt.axis("off")
    plt.imshow(image)
    plt.show()

    
#MNIST#####################################################################################################################################
#Input: 'training' or 'testing' to indicate data set, file-path
#Output: numpy array of floats 0.0-1.0 of rank [-1,28,28,1], integer labels
def read_mnist(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        print("Parameter for read_mnist must be 'testing' or 'training'")
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
        print("MNIST labels loaded from: " + str(path))
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols, 1)
        img = np.array(img, dtype=float) / 255.0
    #get_img = lambda idx: (lbl[idx], img[idx])
    # Create an iterator which returns each image in turn
    #for i in range(len(lbl)):
        #yield get_img(i)
        print("MNIST images loaded from: " + str(path))
        return img, lbl
        
#Input: 3-D numpy array of floats: 0.0-1.0 with 1 channel        
#Output: Display a gray-scale image
def plot_mnist(image):
    #Render a given numpy.uint8 2D array of pixel data.
    from matplotlib import pyplot
    import matplotlib as mpl
    image = image.reshape(28,28)
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image*255, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
