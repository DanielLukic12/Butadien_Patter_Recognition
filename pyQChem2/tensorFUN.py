import tensorflow as tf
import numpy as np

# Input Shape and get Normal Distributad Values in for of shape
def weight_variable(shape):
    # Outputs random values from a truncated normal distribution.
    # stddev ==  deviation of the truncated normal distribution
#    initial = tf.random_normal(shape)
    #initial  = tf.random_uniform(shape, -1, 1)
    initial  = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    

def bias_variable(shape):
#     initial = tf.random_normal(shape)
     #initial = tf.random_uniform(shape, -1, 1)
     initial  = tf.truncated_normal(shape, stddev=0.1)
     return tf.Variable(initial)

def multilayer_perceptron(x, nn_info):

    dimension_Input = nn_info[0]
    dimension_Target = nn_info[np.size(nn_info)-1]
    if np.size(nn_info) == 3:
        layers_info = [nn_info[0]]
    else:
        layers_info = nn_info[1:np.size(nn_info)-2]
    weights = weight_variable([dimension_Input, layers_info[0]])             
    bias = bias_variable([layers_info[0]])
    # Hidden layer with RELU activation
    layer = tf.add(tf.matmul(x, weights), bias)
    #layer = tf.sigmoid(layer)
    lenght_layers = np.size(layers_info)

    if lenght_layers > 1:  
        for i in np.arange(lenght_layers-1):
            weights = weight_variable([layers_info[i], layers_info[i+1]])
            bias = bias_variable([layers_info[i+1]])

            # Hidden layer with activation Function
            layer = tf.add(tf.matmul(layer, weights), bias)
            layer = tf.nn.elu(layer)

            if i == 1 & i == 2 & i == 3 & i == 4:
                    layer = tf.nn.dropout(layer, 0.9)

    # Output layer with linear activation
    weights_out = weight_variable([layers_info[lenght_layers - 1], dimension_Target])
    bias_out = bias_variable([dimension_Target])
    out_layer = tf.matmul(layer, weights_out) + bias_out
    return out_layer


