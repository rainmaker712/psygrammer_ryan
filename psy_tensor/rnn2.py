
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

import numpy as np
from PIL import Image
import datetime,time

now = datetime.datetime(year=2016,month=7,day=7,hour=23,minute=00,second=0,microsecond=0)
dt = datetime.timedelta(minutes=10)

np.random.seed(1)

class Point():
  def __init__(self, x, y):
    self.x = x
    self.y = y

lst = []
for i in range(0, 1251):
  s = now.strftime("%Y%m%d%H%M")
  lst.append("%s" % (s))
  now -= dt

lst.reverse()

checkPt = Point(328, 205)

sx, ex = 300, 340
sy, ey = 185, 225

inputs, targets, maxval = [], [], 50
for name in lst:
  fn = "%s.png" % (name)
  im = Image.open(fn)
  in_data = np.asarray(im, dtype=np.float32)
  data = np.reshape(in_data[sx:ex, sy:ey], (-1))
  inputs.append(data)
  output = np.zeros((maxval))
  output[int(in_data[328,205])-1] = 1.
  targets.append(output)

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables
input_vec_size = lstm_size = len(inputs[0])
time_step_size = 1

batch_size = 1000/8
test_size = 1000/4

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, init_state, lstm_size):
    # X, input shape: (batch_size, input_vec_size, time_step_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (input_vec_size, batch_szie, time_step_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size)
    # XR shape: (input vec_size, batch_size)
    X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    lstm = rnn_cell.GRUCell(lstm_size)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.rnn(lstm, X_split, initial_state=init_state)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

trX, teX = np.asarray(inputs[0:1000]), np.asarray(inputs[1000:1250])
trY, teY = np.asarray(targets[1:1001]), np.asarray(targets[1001:1251])
trX = trX.reshape(-1, input_vec_size, time_step_size)
teX = teX.reshape(-1, input_vec_size, time_step_size)

# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
init_state = tf.placeholder("float", [None, lstm_size])

X = tf.placeholder("float", [None, input_vec_size, time_step_size])
Y = tf.placeholder("float", [None, maxval])

# get lstm_size and output 10 labels
W = init_weights([lstm_size, maxval])
B = init_weights([maxval])

py_x, state_size = model(X, W, B, init_state, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 1.0).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    ntrx = len(trX)
    print "ntrx = ", ntrx
    for i in range(100):
        for start, end in zip(range(0, ntrx, batch_size), range(batch_size, ntrx, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          init_state: np.zeros((batch_size, state_size))})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         init_state: np.zeros((test_size, state_size))})))
