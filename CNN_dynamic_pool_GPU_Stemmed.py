import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
#from collections import Counter
import argparse
import sys
import tempfile
from nltk.stem import PorterStemmer

ps = PorterStemmer()
ngram_embedder = 2
#xtf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None

# SPP- https://github.com/tensorflow/tensorflow/pull/12852/commits/e1868712c0e3966d1ca7237a4ec0ad708f6c9c32


def readData(src):
    datax = pd.read_csv(src)
    datax = datax.dropna()
    datax = datax.reset_index()
    return datax


def pre_processing(text):
    text=str(text).lower()
    text = removeStopWords(text)
    text = re.sub(r'\b[u]\b', 'you', text)
    text = re.sub(r'\b[r]\b', 'are', text)
    text = re.sub(r'\b[y]\b', 'why', text)
    text = re.sub(r'\b[d]\b', 'the', text)
    text = re.sub(r'\b[n]\b', 'and', text)
    text = re.sub(r'\b[m]\b', 'am', text)
    text = re.sub(r'\b[b]\b', 'be', text)
    text = re.sub(r'\b[im]\b', 'i am', text)
    text = re.sub(r'\b([a-z]+[0-9]+|[0-9]+[a-z]+)\w*\b', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r'[.]{2,}', ' . ', text)
    text = re.sub(r'[.]', ' . ', text)
    text = re.sub(r'[!]{2,}', ' ! ', text)
    text = re.sub(r'[!]', ' ! ', text)
    text = re.sub(r'[?]{2,}', ' ? ', text)
    text = re.sub(r'[?]', ' ? ', text)
    text = re.sub(r'[,\-()]', ' ', text)
    text = re.sub(' +', ' ', text)
    tagged_sentence = nltk.tag.pos_tag(text.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    Stemmed = [ps.stem(word) for word in edited_sentence]
    text = ' '.join(Stemmed)
    #text = ' '.join([pprocess.data_loader.stemmer.stem(word) for word in text.split() if word not in pprocess.data_loader.cachedStopWords])
    return(text)


def createVocab(text):
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(text)
    global Vocab
    Vocab = [".", "?", "!"]
    Vocab = Vocab + list(vectorizer.vocabulary_.keys())
    return 0


def embedGram(sentence):
    split_sent = sentence.split()
    if len(split_sent) == 0:
        return np.array([''])
    else:
        output = []
        for i in range(len(split_sent) - ngram_embedder + 1):
            output.append(split_sent[i:i + ngram_embedder])
    if len(output) == 0:
        return np.array([sentence.split()])
    else:
        return np.array(output)


def pad(Tensors):
    max_size = 0
    for t in Tensors:
        if t.shape[1] > max_size:
            max_size = t.shape[1]
    pads = []
    for t in Tensors:
        if t.shape[1] < max_size:
            zeroPad = np.zeros(shape=(t.shape[0], max_size - t.shape[1]))
            t = np.concatenate((t, zeroPad), axis=1)
            pads.append(t)
    return pads


def createTensors(feedback):
    embedded_grams = embedGram(feedback)
    t = pd.DataFrame(np.zeros(shape=(len(Vocab), len(embedded_grams))))
    t = t.astype(int)
    for i in range(t.shape[1]):
        for word in embedded_grams[i]:
            if word in Vocab:
                t.iloc[Vocab.index(word)][i] = 1
    t = t.values
    return t


def removeStopWords(text):
    stops = set(stopwords.words('english'))
    t = []
    for w in text.split():
        if w.lower() not in stops:
            t.append(w)
    text = ' '.join(t)
    return text


def spatial_pyramid_pool(inputs, dimensions=[2,1], mode='max', implementation='kaiming'):
    pool_list = []
    if implementation == 'kaiming':
        for pool_dim in dimensions:
            pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, mode)
    else:
        shape = inputs.get_shape().as_list()
        for d in dimensions:
            h = shape[1]
            w = shape[2]
            ph = np.ceil(h * 1.0 / d).astype(np.int32)
            pw = np.ceil(w * 1.0 / d).astype(np.int32)
            sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
            sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
            pool_result = tf.nn.max_pool(inputs,
                                         ksize=[1, ph, pw, 1],
                                         strides=[1, sh, sw, 1],
                                         padding='SAME')
            pool_list.append(tf.reshape(pool_result, [tf.shape(inputs)[0], -1]))
    r1 = tf.reduce_max(pool_list, axis=0)
    print('r1 shape:')
    print(r1.get_shape())
    return tf.reduce_mean(r1, axis=1)


def max_pool_2d_nxn_regions(inputs, output_size: int, mode: str):
    inputs_shape = tf.shape(inputs)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)

    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))

    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            #print("col:"+str(col))
            start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            #print("start_h:" + str(sess.run(start_h)))
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            #print("end_h:" + str(sess.run(end_h)))
            start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            #print("start_w:" + str(sess.run(start_w)))
            end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            #print("end_w:" + str(sess.run(end_w)))

            #pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=0)
            result.append(pool_result)
    return result


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  ''' 
  '''
  with tf.name_scope('reshape'):
      print('reshape x:')
      #print(x.shape)
      x_image = tf.cast(tf.reshape(x, [1, 3614, -1, 1]), dtype='float32')
      print('reshape x_image:')
      print(x_image.get_shape().as_list())

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    print('conv1:')
    print(x_image.get_shape().as_list())
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    print('pool1:')
    print(h_conv1.get_shape().as_list())
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    print('conv2:')
    print(h_pool1.get_shape().as_list())
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Final pooling layer.
  with tf.name_scope('spp'):
    h_poolF = spatial_pyramid_pool(h_conv2, dimensions=[2, 2])

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.device("/gpu:0"):
        W_fc1 = weight_variable([2556 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_poolF, [-1, 2556 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.device("/gpu:0"):
    W_fc2 = weight_variable([1024, 15])
    b_fc2 = bias_variable([15])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return data_shuffle, labels_shuffle


def dv_one_hot(target_data):
    uniques = list(set(target_data))
    d = {x:y for x,y in zip(uniques, range(len(uniques)))}
    num_target_data = np.array(list(map(d.get, target_data)))
    one_hot = np.zeros([len(target_data), len(uniques)])
    one_hot[np.arange(len(target_data)), num_target_data] = 1
    return one_hot


def main():
    src = "Myntra-data-Category-fullpath_spell_corrected.csv"
    datax = readData(src)
    feedbacks = [pre_processing(x) for x in datax["Corrected"]]
    feedbacks = [removeStopWords(x) for x in feedbacks]
    createVocab(feedbacks)

    Tensors = []
    for feedback in feedbacks:
        Tensors.append(createTensors(feedback))

    target_data = np.asarray(datax["Sub Dept 2"], dtype=np.unicode_)
    feature_data = Tensors
    target_data = [dv.replace('reverse logistics_Pickup time', 'reverse logistics_Pickup Time') for dv in target_data]
    target_data = [dv.replace('Reverse Logistics_Pickup Time', 'reverse logistics_Pickup Time') for dv in target_data]
    target_data = [dv.replace('reverse logistics_pickup time', 'reverse logistics_Pickup Time') for dv in target_data]
    target_data = [dv.replace('Delivery_Delivery Speed', 'Delivery_Delivery speed') for dv in target_data]
    target_data = [dv.replace('delivery_Delivery Communications / Updates', 'Delivery_Delivery Communications / Updates') for dv in
        target_data]
    target_data = [dv.replace('Retail Revenue_pricing', 'Retail Revenue_Pricing') for dv in target_data]
    target_data = [dv.replace('Retail Revenue_offer & discount', 'Retail Revenue_Offer & Discount') for dv in
                   target_data]

    int_target_data = dv_one_hot(target_data)

    X_train, X_test, y_train, y_test = train_test_split(feature_data, int_target_data, test_size=0.3, random_state=0)

    # Create the model
    x = tf.placeholder(tf.float32, [1, 5109, None])

    # Define loss and optimizer
    y_= tf.placeholder(tf.float32, [1, 15])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_accuracy = 0
        test_accuracy = 0
        for i in range(20000):
            batch = next_batch(1, X_train, y_train)
            temp_y = batch[1][0]
            temp_x = batch[0][0]
            y_feed = temp_y.reshape(1, temp_y.shape[0])
            x_feed = temp_x.reshape(1, temp_x.shape[0], temp_x.shape[1])

            train_accuracy += accuracy.eval(feed_dict={x: x_feed, y_: y_feed, keep_prob: 1.0})
            # accuracy_train_list = accuracy_train_list.append(train_accuracy)
            '''if i % 100 == 0:
                # print('step %d, training accuracy %g' % (i,sum(accuracy_train_list)/len(accuracy_train_list)))
                print('step %d, training accuracy %g' % (i, train_accuracy / i))
            '''
       
            # train model
            train_step.run(feed_dict={x: x_feed, y_: y_feed, keep_prob: 0.5})

        for j in range(y_test.shape[0]):
            batch2 = next_batch(1, X_test, y_test)
            temp_y = batch2[1][0]
            temp_x = batch2[0][0]
            y_feed_t = temp_y.reshape(1, temp_y.shape[0])
            x_feed_t = temp_x.reshape(1, temp_x.shape[0], temp_x.shape[1])
            test_accuracy += accuracy.eval(feed_dict={x: x_feed_t, y_: y_feed_t, keep_prob: 1.0})
            # accuracy_test_list = accuracy_test_list.append(test_accuracy)

        #print('%d test accuracy %g' % (j, test_accuracy/j))
        tr_a = train_accuracy/i
        te_a = test_accuracy/j
        with open('Accuracy_logger.txt', 'a') as the_file:
             the_file.write('Train:'+ str(tr_a))
             the_file.write('\n')
             the_file.write('Test:'+str(te_a))
             the_file.write('\n')
             the_file.write('\n')


if  __name__ =='__main__':
    main()
