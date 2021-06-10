import pandas as pd
from sklearn import model_selection
import stellargraph as sg
import tensorflow as tf
import  scipy.sparse as sp
import  numpy as np

from grapher import arr

print(arr)

def load_my_data(file_name):
    # your own code to load data into Pandas DataFrames, e.g. from CSV files or a database
    ...
    adj = sp.load_npz(file_name+"_adj.npz")
    features = np.load(file_name+"_feature.npy", allow_pickle=True)
    labels = np.load(file_name+"_label.npy",allow_pickle=True)


nodes, edges, targets = load_my_data()

# Use scikit-learn to compute training and test sets
train_targets, test_targets = model_selection.train_test_split(targets, train_size=0.5)


###########################################################################################

# convert the raw data into StellarGraph's graph format for faster operations
graph = sg.StellarGraph(nodes, edges)

generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")

# two layers of GCN, each with hidden dimension 16
gcn = sg.layer.GCN(layer_sizes=[16, 16], generator=generator)
x_inp, x_out = gcn.in_out_tensors() # create the input and output TensorFlow tensors

# use TensorFlow Keras to add a layer to compute the (one-hot) predictions
predictions = tf.keras.layers.Dense(units=4, activation="softmax")(x_out)

# use the input and output tensors to create a TensorFlow Keras model
model = tf.keras.Model(inputs=x_inp, outputs=predictions)

###########################################################################################

# prepare the model for training with the Adam optimiser and an appropriate loss function
model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])



# train the model on the train set
model.fit(generator.flow(train_targets.index, train_targets), epochs=5)

# check model generalisation on the test set
(loss, accuracy) = model.evaluate(generator.flow(test_targets.index, test_targets))
print(f"Test set: loss = {loss}, accuracy = {accuracy}")
