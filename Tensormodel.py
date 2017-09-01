#More intuitive functions to be implemented in the Tensorflow model. Meant to explain how tensorflow ops work as well
#Use in an interactive session

import tensorflow as tf

#Variable def##########################################################################################################################

#Returns a matrix of weights with the standard deviation: sdev. All values are within 2 standard devs of 0
def weight_variable(shape, sdev):
    initial = tf.truncated_normal(shape, stddev=sdev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Convolution functions####################################################################################################################
#Input: 3-D Tensors: input tensor, filter tensor and strides array
#Output: A 1-D convolution 
def conv_1d(x, W, strides):
    return tf.nn.conv1d(x, W, strides=strides, padding='SAME')

#Input: 4-D Tensors: x has dim [batch, in_height, in_width, in_channels]  [filter_height, filter_width, in_channels, out_channels]
#Output: A tensor with shape=[images, out_height, out_width, filter_height * filter_width * in_channels], out_width and height depend on strides
def conv_2d(x, W, strides):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

#Input: 5-D Tensors: input tensor, filter tensor and strides array
#Output: A 4-D convolution
def conv_3d(x, W, strides):
    tf.nn.conv3d(x, W, strides=strides, padding='SAME')

#Pooling Functions##################################################################################################################
#Input: A 4D tensor
#Output: The poolings of the tensor with 2x2 chuncks and strides of 2. Only pools in 2Common poolings for images
def maxpool2d_size2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Input: A 4d tensor, pooling ratio: a vector ex: [1.0, 1.44, 1.73, 1.0]. The first and last elements must be 1.0. The height_ratio and width_ratio correspond to the ratio of the sizes of the input / output
#Output: The max-poolings tensor. The shape will be reduced by the given ratios
def fractional_max_pooling_layer(x, height_ratio, width_ratio):
    return tf.nn.fractional_max_pool(x, [1.0, height_ratio, width_ratio, 1.0])

#Activaton Funcs###################################################################################################################
#Exponential linear
def exp_Activation(x):
    return tf.nn.elu(x, name=None)

def ReLU_Activation(x):
    return tf.nn.relu(x)

###################################################################################################################################

#Input: A 4-D input_tensor, the number of input features in each channel of the input_tensor, the number of output features
#Processing: Reshape each data item in the input_tensor to a 1-D vector, Mult w/ a 2D weight to obtain a 1D vector of length: num_output_feat,  add biases
#Output: A 1-D vector for each data item
def Fully_connected_layer(input_tensor, num_input_feat, num_output_feat):
    W_fc = weight_variable([num_input_feat, num_output_feat])
    b_fc = bias_variable([num_output_feat])
    input_pool_flat = tf.reshape(input_tensor, [-1 , num_input_feat])
    h_fc = tf.nn.relu(tf.matmul(input_pool_flat, W_fc) + b_fc)
    return h_fc

#Apply dropout
def Apply_dropout(input_tensor, keep_prob):
    return tf.nn.dropout(input_tensor, keep_prob)

#Input:
#Processing:
#Output: 
def Readout_layer(input_tensor, num_input_feat, num_output_feat):
    W_fc = weight_variable([num_input_feat, num_output_feat])
    b_fc = bias_variable([num_output_feat])
    y_conv = tf.matmul(input_tensor, W_fc) + b_fc
    return y_conv

def Gradient_Descent_Optimize_Network(labels, y_conv):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, labels))
    Optimize = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal( tf.argmax(labels, 1),tf.argmax(y_conv, 1) ), tf.float32))
    return Optimize, cross_entropy, accuracy

#Input: <tensorflow var>, one-hot-labels and the un-normalized output of the neural-net
#Optimize network, make predictions, and return class-confidences, and <int> class predictions, <int> labels and accuracy
def Adam_Optimize_Network(labels, y_conv):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, labels))
    Optimize = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal( tf.argmax(labels, 1),tf.argmax(y_conv, 1) ), tf.float32))
    return Optimize, cross_entropy, accuracy

#For each data item get the confidence probability for all classes
def Normalized_predictions(y_conv):
    y_normal_confidence = tf.nn.softmax(y_conv)
    return y_normal_confidence

#Returns the numeric predictions or the numeric classes
def Reduce_one_hot(y_norm):
    return tf.argmax(y_norm, 1)

##########################################################################################################################################
#IMAGE FUNCTIONS##########################################################################################################################

#Input: an image, the size of the crop square and number of channels
#Output: Cropped image
def crop_image(image, img_size_cropped, num_channels):
    image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
    return image

#Flips an image about the diagonal (top-left to bottom-right) axis
def Image_transpose(image):
    image = tf.image.transpose_image(image)
    return image

#Input: A grayscale image w/ dimensions: [height, width, 1]
#Output: An RGB image w/ dimensions: [height, width, 3]
def Image_grayscale_to_rgb(image):
    image = tf.image.grayscale_to_rgb(image, name=None)
    return image

#Input: An RGB image w/ dimensions: [height, width, 3]
#Output: A grayscale image w/ dimensions: [height, width, 1]
def Image_rgb_to_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image


#Image Pre-processing#####################################################################################################################
#Input: an image, a scalar int representing the number of rotations CCW by 90 degrees
#Output: The image rotated
def Image_rotate(image, num_rot):
    image = tf.image.rot90(image, k=1, name=None)
    return image

#Input: A single RGB image: [height, width, 3]
#Output: The RGB image with randomly adjusted hue, contrast, brightness, saturation
def Image_random_rgb_distort(image):
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=.125)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #Make sure color levels remain in the range [0, 1]
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image

#Input: Any image: [height, width, channels]
#Output: The same image with a 50% chance of being mirrored vertically and 50% horizontally
def Image_random_flip_distort(image):
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    return image

#Randomly flip and adjust each of the input 
def pre_process_images(images, process_images):
    # Use TensorFlow to loop over all the input images
    #tf.map_fn: Unpack all images along dimension: 0
    #An optional 3rd parameter holds the max number of iterations allowed to run in parallel (default=10)
    def f1(): return images
    def f2(): return tf.map_fn(lambda image: Image_random_rgb_distort(image), images)
    def f3(): return tf.map_fn(lambda image: Image_random_flip_distort(image), images)
    images = tf.cond(process_images, f2, f1)
    images = tf.cond(process_images, f3, f1)
    return images

#Input: 4-D Tensor of shape [batch, height, width, channels] or 3-D tensor of shape [height, width, channels],  A 1-D int32 Tensor of 2 elements: new_height, new_width
#Output: Each image is resized to: [new_height, new_width, channels] 
def Resize_images_to(images, new_shape):
    images = tf.image.resize_images(images, new_shape)
    return images

#Input: 4-D Tensor of shape [batch, height, width, channels] or 3-D tensor of shape [height, width, channels],  [target_height,target_width]
#Output: Image resized by cropping or padding
def Resize_images_crop_pad(images, new_shape):
    images = tf.image.resize_images(images, new_shape)
    return images

#Get files###############################################################################################################################
#Input: 0-D string. The encoded image bytes, target ouput color channels
#Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate operation to convert the input bytes string into a Tensor of type uint8.
#Output: gifs return a tensor of dim: [num_frames, height, width, 3]; PNG, JPEG dim: [height, width, num_channels]
def Decode_image_from_bytes(contents, output_color_channels=None):
    image = tf.image.decode_image(contents, channels=output_color_channels)
    return image

def Encode_image_as_jpeg(image):
    return tf.image.encode_jpeg(image)
        
def Encode_image_as_png(image):
    return tf.image.encode_png(image)