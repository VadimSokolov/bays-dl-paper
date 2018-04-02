#!/usr/bin/env python

import numpy as np

# Neural networks take input as vectors so we have to convert integers to vectors using one-hot encoding
# This function will encode a given integer sequence into RNN compatible format (one-hot representation)

def encode(X,seq_len, vocab_size):
    x = np.zeros((len(X),seq_len, vocab_size), dtype=np.float32)
    for ind,batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[ind, j, elem] = 1
    return x


# This is a generator function which can generate infinite-stream of inputs for training

def batch_gen(batch_size=32, seq_len=10, max_no=100):
    # Randomly generate a batch of integer sequences (X) and its sorted
    # counterpart (Y)
    x = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)
    y = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)

    while True:
	# Generates a batch of input
        X = np.random.randint(max_no, size=(batch_size, seq_len))

        Y = np.sort(X, axis=1)

        for ind,batch in enumerate(X):
            for j, elem in enumerate(batch):
                x[ind, j, elem] = 1

        for ind,batch in enumerate(Y):
            for j, elem in enumerate(batch):
                y[ind, j, elem] = 1

        yield x, y
        x.fill(0.0)
        y.fill(0.0)

from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dropout, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers import recurrent
import numpy as np
# from data import batch_gen, encode
RNN = recurrent.LSTM

# global parameters.
batch_size=32
seq_len = 10
max_no = 100

# Initializing model 
model = Sequential()

# This is encoder RNN (we are taking a variant of RNN called LSTM, because plain RNN's suffer from long-term dependencies issues
model.add(RNN(100, input_shape=(seq_len, max_no)))

# Dropout to enhace RNN's generalization capacities 
model.add(Dropout(0.25))

# At this point RNN will generate a summary vector of the sequence, so to feed it to decoder we need to repeat it lenght of output seq. number of times
model.add(RepeatVector(seq_len))

# Decoder RNN, which will return output sequence 
model.add(RNN(100, return_sequences=True))

# Adding linear layer at each time step 
model.add(TimeDistributed(Dense(max_no)))

# Adding non-linearity on top of linear layer at each time-step, since output at each time step is supposed to be probability distribution over max. no of integer in sequence
# we add softmax non-linearity
model.add(Dropout(0.5))
model.add(Activation('softmax'))

# Since this is a multiclass classification task, crossentropy loss is being used. Optimizer is adam, which is a particular instance of adaptive learning rate Gradient Descent methods
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Now the training loop, we'll sample input batches from the generator function written previously and feed it to the RNN for learning

for ind,(X,Y) in enumerate(batch_gen(batch_size, seq_len, max_no)):
	loss, acc = model.train_on_batch(X, Y)
	# We'll test RNN after each 250 iteration to check how well it is performing
	if ind % 250 == 0:
		testX = np.random.randint(max_no, size=(1, seq_len))
		test = encode(testX, seq_len, max_no)
		print testX
		#pdb.set_trace()
		y = model.predict(test, batch_size=1)
		print "actual sorted output is"
		print np.sort(testX)
		print "sorting done by RNN is"
		print np.argmax(y, axis=2)
		print "\n"
