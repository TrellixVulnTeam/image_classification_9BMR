import pickle
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile
import helper
import numpy as np

cifar10_dataset_folder_path = 'cifar-10-batches-py'

floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)






batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)




def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    a = 0.1
    b = 0.9
    layer_min = 0
    layer_max = 255
    return a + (((x - layer_min)*(b - a) )/( layer_max - layer_min ) )
    

tests.test_normalize(normalize)





encoder = LabelBinarizer()
encoder.fit(range(10))
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return encoder.transform(x).astype(np.float32)

tests.test_one_hot_encode(one_hot_encode)




helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)





valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))





def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    
    return tf.placeholder(tf.float32, shape=((None,) + image_shape), name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)




def one_pad(tup):
    return ((1,) + (tup) + (1,))
    

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    
    input_depth = x_tensor.shape[3].value
   
    weights_size = (conv_ksize[0], conv_ksize[1], input_depth, conv_num_outputs)
    weights = tf.Variable(tf.truncated_normal(weights_size, mean=.0, stddev=.01))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    
    conv = tf.nn.conv2d(x_tensor, weights, one_pad(conv_strides), padding='SAME')
    conv = tf.nn.bias_add(conv, bias)
    conv = tf.nn.relu(conv)

    return tf.nn.max_pool(conv, one_pad(pool_ksize), one_pad(pool_strides), padding='SAME') 


tests.test_con_pool(conv2d_maxpool)




import operator
from functools import reduce

def size_of_flatten(x_tensor):
    sizes = [size.value for size in x_tensor.shape[1:]]
    
    return reduce(operator.mul, sizes, 1)

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    
    return tf.reshape(x_tensor, shape=(-1, size_of_flatten(x_tensor)))


tests.test_flatten(flatten)




def fully_conn(x_tensor, num_outputs):
    
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    num_inputs = x_tensor.shape[1].value
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], mean=.0, stddev=.01))
    
    bias = tf.Variable(tf.zeros(shape=[num_outputs]))
    
    return tf.nn.relu(tf.nn.bias_add(tf.matmul(x_tensor, weights), bias))


tests.test_fully_conn(fully_conn)




def output(x_tensor, num_outputs):
    
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    num_inputs = x_tensor.get_shape().as_list()[-1]
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], mean=.0, stddev=.01))
    
    bias = tf.Variable(tf.zeros(shape=[num_outputs]))
    
    return tf.nn.bias_add(tf.matmul(x_tensor, weights), bias)


tests.test_output(output)




def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    conv = conv2d_maxpool(
        x_tensor=x, 
        conv_num_outputs=48,
        conv_ksize=(4,4),
        conv_strides=(1,1),
        pool_ksize=(2,2), 
        pool_strides=(2,2))
    
    conv = conv2d_maxpool(
        x_tensor=conv, 
        conv_num_outputs=256,
        conv_ksize=(4,4),
        conv_strides=(1,1),
        pool_ksize=(2,2), 
        pool_strides=(2,2))
    
    conv = conv2d_maxpool(
        x_tensor=conv,
        conv_num_outputs=512,
        conv_ksize=(4,4),
        conv_strides=(1,1),
        pool_ksize=(2,2), 
        pool_strides=(2,2))
    
    flattened = tf.nn.dropout(flatten(conv), keep_prob)
    

    hidden_layer = fully_conn(flattened, 512)
    hidden_layer = fully_conn(hidden_layer, 256) 
    
    return output(hidden_layer, 10)




tf.reset_default_graph()

x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

logits = conv_net(x, keep_prob)

logits = tf.identity(logits, name='logits')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)




def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    return session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})


tests.test_train_nn(train_neural_network)




def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    accuracy_val = session.run(accuracy, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    cost_val = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    
    print('Accuracy: {} - Cost {}'.format(
                accuracy_val,
                cost_val))
    
    validation_accuracy_val = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    validation_cost_val = session.run(cost, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    
    print('Validation Accuracy: {} - Cost {}'.format(
                validation_accuracy_val,
                validation_cost_val))




epochs = 20
batch_size = 512
keep_probability = 0.6

print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)

save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)




import tensorflow as tf
import pickle
import helper
import random

try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
