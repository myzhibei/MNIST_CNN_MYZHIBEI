from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import sys
import os
import time
import urllib
import numpy

import tensorflow as tf

# 文件路径
log_path = r"./logs"  # 日志文件路径
dataset_path = r"./data"  # 数据集存放路径
model_save_path = r"./model"  # 模型待存储路径

# Images source and details
SOURCE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
WORK_DIRECTORY = dataset_path
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.

# Train/ Eval Params
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32


# Helper functions used to download the data set and evaluate the results


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32


def maybe_download(filename):
    """Download the data from source, unless it's already here."""
    if not tf.io.gfile.exists(WORK_DIRECTORY):
        tf.io.gfile.makedirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            SOURCE_URL + filename, filepath)
    with tf.io.gfile.GFile(filepath) as f:
        size = f.size()

    print("Successfully downloaded", filename, size, "bytes.")
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)  # 每个像素存储在文件中的大小为16bits
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE *
                              num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        # 像素值[0, 255]被调整到[-0.5, 0.5]
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        # 调整为4 维张量[image index, y, x, channels]
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


# 加载标签：
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)  # 每个标签存储在文件中的大小为8bits
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 * numpy.sum(numpy.argmax(predictions, 1) ==
                          labels) / predictions.shape[0]
    )


# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction, feed_dict={eval_data: data[begin:end, ...]}
            )
        else:
            batch_predictions = sess.run(
                eval_prediction, feed_dict={
                    eval_data: data[-EVAL_BATCH_SIZE:, ...]}
            )
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


# 初始化变量：
#  下面的变量包含所有可训练的权重。当我们调用时将分配它们时，它们被传递一个初始值：
# {tf.global_variables_initializer().run()}
# 5x5 filter, depth 32.
conv1_weights = tf.Variable(
    tf.compat.v1.random.truncated_normal(
        [5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=data_type()
    )
)
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(
    tf.compat.v1.random.truncated_normal(
        [5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type()
    )
)
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
# fully connected, depth 512.
fc1_weights = tf.Variable(
    tf.compat.v1.random.truncated_normal(
        [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
        stddev=0.1,
        seed=SEED,
        dtype=data_type(),
    )
)
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
fc2_weights = tf.Variable(
    tf.compat.v1.random.truncated_normal(
        [512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type()
    )
)
fc2_biases = tf.Variable(tf.constant(
    0.1, shape=[NUM_LABELS], dtype=data_type()))


# NN
def model(data, train=False):
    """The Model definition."""
    # 2D 卷积，带有“SAME”填充（即输出要素图与输入的大小相同）。
    # 请注意，{strides}是一个4D 数组，其形状与数据布局匹配：[image index，y，x，depth]。
    conv = tf.nn.conv2d(data, conv1_weights, strides=[
                        1, 1, 1, 1], padding="SAME")
    #  偏置和ReLU 非线性激活。
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(
        relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )
    conv = tf.nn.conv2d(pool, conv2_weights, strides=[
                        1, 1, 1, 1], padding="SAME")
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    #  最大池化。
    #  内核大小规范{ksize}也遵循数据布局。  这里我们有一个2 的池化窗口和2 的步幅。
    pool = tf.nn.max_pool(
        relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )
    #  将特征图变换为2D 矩阵，以将其提供给完全连接的图层。
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
    )
    #  全连接层。 Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


# CNN 模型构建：
# Import data
train_data_filename = maybe_download("train-images-idx3-ubyte.gz")
train_labels_filename = maybe_download("train-labels-idx1-ubyte.gz")
test_data_filename = maybe_download("t10k-images-idx3-ubyte.gz")
test_labels_filename = maybe_download("t10k-labels-idx1-ubyte.gz")

# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

# Generate a validation set.
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]
num_epochs = NUM_EPOCHS
train_size = train_labels.shape[0]

# 创建输入占位符：
# 这是训练样本和标签被送到图表的地方。
# 这些占位符节点将在每个节点输入一批训练数据
# 训练步骤使用{feed_dict}参数进行下面的Run（）调用。
train_data_node = tf.compat.v1.placeholder(
    data_type(), shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
)
train_labels_node = tf.compat.v1.placeholder(tf.int64, shape=(BATCH_SIZE,))
eval_data = tf.compat.v1.placeholder(
    data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
)

# 训练CNN

# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, True)
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits
    )
)

# L2 regularization for the fully connected parameters.
regularizers = (
    tf.nn.l2_loss(fc1_weights)
    + tf.nn.l2_loss(fc1_biases)
    + tf.nn.l2_loss(fc2_weights)
    + tf.nn.l2_loss(fc2_biases)
)
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype=data_type())
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
    0.01,  # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    train_size,  # Decay step.
    0.95,  # Decay rate.
    staircase=True,
)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(
    loss, global_step=batch
)

# Predictions for the current training minibatch.
train_prediction = tf.nn.softmax(logits)

# Predictions for the test and validation, which we'll compute less often.
eval_prediction = tf.nn.softmax(model(eval_data))

# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print("Initialized!")
    # Loop through training steps.
    for step in range(int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset: (offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset: (offset + BATCH_SIZE)]

        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}

        # Run the optimizer to update weights.
        sess.run(optimizer, feed_dict=feed_dict)
        # print some extra information once reach the evaluation frequency

        if step % EVAL_FREQUENCY == 0:
            # fetch some extra nodes' data
            l, lr, predictions = sess.run(
                [loss, learning_rate, train_prediction], feed_dict=feed_dict
            )
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(
                "Step %d (epoch %.2f), %.1f ms"
                % (
                    step,
                    float(step) * BATCH_SIZE / train_size,
                    1000 * elapsed_time / EVAL_FREQUENCY,
                )
            )
            print("Minibatch loss: %.3f, learning rate: %.6f" % (l, lr))
            print("Minibatch error: %.1f%%" %
                  error_rate(predictions, batch_labels))
            print(
                "Validation error: %.1f%%"
                % error_rate(eval_in_batches(validation_data, sess), validation_labels)
            )
            sys.stdout.flush()

    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print("Test error: %.1f%%" % test_error)
