# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:25:24 2017

@author: lukic
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import pyQChem2.start as qc
from pyQChem2.create_data_set import xyz2zmat, normalize_data
import helpers.tensorfunctions as tfFun
import os
import numpy as np
import matplotlib.animation as animation
import random
import helpers.plotting as plotting

project_path = os.path.dirname(os.path.abspath(__file__))

# Define nodes of Layers
n_input = 24
nn_hidden1 = 20
nn_hidden2 = 20
nn_hidden3 = 10
n_classes = 1
configurations = []
neuralnet_info = [n_input, 300, 300, 300, 300, n_classes]
configurations.append(neuralnet_info)
regularization = True

# Define hyperparameters
beta_1 = 0.9
beta_2 = 0.999
learning_rate = 0.001
epoch = 2000
batch_size = 100
scale_l2_regul = 0.0000001

display_step = 10
size_disp_test = 50
split_percent = 90

# Ipmort input and target data
aimd_out = qc.read("pyQChem2/aimd.out", silent=True)
aimd_job1 = aimd_out.list_of_jobs[0]
aimd_job2 = aimd_out.list_of_jobs[1]

# Get geometries and energies
trajectory_geometries = aimd_job2.aimd.geometries
trajectory_energies = [aimd_job1.general.energy] + aimd_job2.aimd.energies

length_dataset = len(trajectory_geometries)

# Normalize angles and dihedral angles of input data
input_data = np.zeros((n_input, length_dataset))
for i, geo in enumerate(trajectory_geometries):
    input_data[:, i] = normalize_data(xyz2zmat(geo))

# Get input data unnormalized
input_data_un = np.zeros((n_input, length_dataset))
for i, geo in enumerate(trajectory_geometries):
    input_data_un[:, i] = xyz2zmat(geo)

# Calc mean and std of input data
mean_input = np.mean(input_data, axis=1)
std_input = np.std(input_data, axis=1)

# Normalize all input data
input_data = input_data - np.tile(mean_input, (length_dataset, 1)).T
input_data /= np.tile(std_input, (length_dataset, 1)).T

# Remove first two input data
input_data = input_data[:, 2:-1]

# Define variable to randomize data data
shuffle_ind = list(range(input_data.shape[1]))
random.shuffle(shuffle_ind)

# Randomize order of input data
input_data = input_data[:, shuffle_ind]

# Split in Training and Test Data
size_data = input_data.shape[1]
size_train = int(np.round((size_data/100)*split_percent, 0))
size_test = size_data - 1 - size_train

# Plotting input data normalized and unnormalized
plotting.plot_input(input_data_un=input_data_un, input_data=input_data,
                    show=False)

# Define training and test input dataset
training_input = input_data[:, 0:size_train].T
test_input = input_data[:, size_train + 1:size_data].T

# Min and max for normalization
trajgeo_min = np.min(trajectory_energies)
trajgeo_max = np.max(trajectory_energies)

# Normalize target data between -1 and 1
normalized_target = 2*(trajectory_energies - trajgeo_min)\
                      / (trajgeo_max - trajgeo_min) - 1
# Define variable to calculate back from norm
remove_norm = (trajgeo_max - trajgeo_min)/2

normalized_target = normalized_target[2:size_data + 2]

# Plot target data first 200
plotting.plot_target(target_to_plot=normalized_target, show=False)

# Plotting target data normalized in a histogram
plotting.plot_hist(target_data=normalized_target, show=False)

# Randomize target data
normalized_target = normalized_target[shuffle_ind]

# Define training and test target dataset
training_target = normalized_target[0:size_train].reshape((size_train, 1))
test_target = normalized_target[size_train + 1:size_data]


# for config in configurations:
# tf Graph input
inputs = tf.placeholder(tf.float32, [None, n_input])
# tf Graph output
outputs = tf.placeholder(tf.float32, [None, n_classes])

# Build together neural net
pred = tfFun.multilayer_perceptron(inputs, neuralnet_info,
                                   drop_out=regularization)
global_step = tf.Variable(0, trainable=False)
# Define decaying learning rate
if regularization:
    starter_learning_rate = learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step, 400, 0.96,
                                               staircase=True)

# Get all trainable variables for l2 regularization
weights = tf.trainable_variables()

# Define l2 regularizer
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=scale_l2_regul,
                                                  scope=None)
regularization_penalty_l2 = tf.contrib.layers\
                            .apply_regularization(l2_regularizer, weights)

# Define cost function with l2 regularization
if regularization:
    cost = tf.losses.mean_squared_error(outputs,
                                        pred) + regularization_penalty_l2
else:
    cost = tf.losses.mean_squared_error(outputs,
                                        pred)
# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)\
                                  .minimize(cost, global_step=global_step)

# Get global variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# List for animation
results_anim = []
# Define name and path to save results
result_path = project_path + "/results/"
path_file = result_path

# Make folder for results if not existing
if not os.path.exists(result_path):
    os.makedirs(result_path)

batch_count = int(round(np.size(training_target)/batch_size))

with tf.Session() as sess:
    # Run session with global variables
    sess.run(init)

    # List for training and test error
    cost_train_arr = []
    cost_test_arr = []

    # Start training for each epoch
    for e in range(epoch):
        cost_err_arr = []
        # Shuffle data for each epoch
        shuffle_ind = list(range(training_input.shape[0]))
        random.shuffle(shuffle_ind)

        shuffle_order = list(range(input_data.shape[0]))
        random.shuffle(shuffle_order)

#        training_input = training_input[:, shuffle_order]
        training_input = training_input[shuffle_ind, :]
        training_target = training_target[shuffle_ind]

        # Start batch training
        for batch in range(batch_count):
            batch_from = batch*batch_size
            batch_to = (batch + 1)*batch_size - 1
            input_training = training_input[batch_from:batch_to, :]
            target_training = training_target[batch_from:batch_to]

            feed_dict = {inputs: input_training, outputs: target_training}
            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
            # Cost error for batches
            cost_err_arr.append(c)

            # Print Cost error for display step
        if e % display_step == 0:
            print('Epoch: ', str(e), '   Cost per epoch: ',
                  str(np.sum(cost_err_arr)/(np.size(cost_err_arr))))

        # Get training error and add to list
        cost_train = sess.run(cost, feed_dict={inputs: training_input, outputs:
                              training_target.reshape((size_train, 1))})
        cost_train_arr.append(cost_train)

        cost_test = sess.run(cost, feed_dict={inputs: test_input, outputs:
                             test_target.reshape((size_test, 1))})
        cost_test_arr.append(cost_test)

        # Save training and test result in list
        size_from = size_train - size_disp_test
        size_to = size_train + size_disp_test
        test_input_extract = input_data[:, size_from: size_to].T
        test_output_extract = normalized_target[size_from:size_to].reshape(
                (size_disp_test*2, 1))

        test_result = sess.run(pred, feed_dict={inputs: test_input_extract,
                                                outputs: test_output_extract})

        # Save unnormalized test result in list for animation
        test_result_un = (test_result + 1)*remove_norm + trajgeo_min
        results_anim.append(np.reshape(test_result_un, size_disp_test*2))

        # Covergence criteria for traning
        if e > 1:
            if np.abs(cost_test_arr[e] - cost_test_arr[e - 1]) < 0.00000001:
                break

    # Plot reference and tested data points.
    reference_data = (normalized_target[size_from: size_to] + 1
                      )*remove_norm + trajgeo_min
    feed_dict = {inputs: test_input_extract, outputs: test_output_extract}
    prediction_test = sess.run(pred, feed_dict=feed_dict)
    tested_data = (prediction_test + 1)*remove_norm + trajgeo_min

    plotting.plot_tested(reference_data=reference_data, prediction=tested_data,
                         show=True, path_to_save=path_file)

    # Get all predicted test targets
    test_pred = sess.run(pred, feed_dict={
            inputs: test_input, outputs: test_target.
            reshape((size_test, 1))})

    # Caclulate back from normalization
    test_target_un = (test_target + 1)*remove_norm + trajgeo_min
    test_pred_un = (test_pred + 1)*remove_norm + trajgeo_min

    # Plotting reference data against prediction data not normalized
    plotting.plot_against(reference=test_target_un, prediction=test_pred_un,
                          show=True, path_to_save=path_file)

    cost_test_sqrd = np.sqrt(cost_test_arr[-1])
    print('------------------------------------------')
    print('Costs for configuration')
    print(neuralnet_info)
    print("Cost Test Error Root= "+str(cost_test_sqrd) +
          "Max Epoch= " + str(e))
    print("Cost Test Error= "+str(cost_test_arr[-1]) + "Max Epoch= " + str(e))

    # Save the variables to disk.
    save_path = saver.save(sess, path_file + "model.ckpt")


# Print extraction of reference and prediction data
print('Prediction')
print(test_target_un[200:210][:])
print('Reference')
print(test_pred_un[200:210])

# Set up plotting for animation
fig_anim = plt.figure()
ax = plt.axes()
line, = plt.plot(results_anim[0], 'bx-')
plot_data = (normalized_target[size_from:size_to] + 1)*remove_norm\
            + trajgeo_min
line2, = plt.plot(plot_data, 'ro-')

# Plot training and test error
plotting.plot_errors(training_error=cost_train_arr, test_error=cost_test_arr,
                     show=True, path_to_save=path_file, epoch=e)

title = ax.text(4, 2.5, "")


# Animation function
def animate(i):
    title.set_text("Epoch: " + str(i))
    line.set_ydata(results_anim[i])
    return line, line2

# Save animation
anim = animation.FuncAnimation(fig_anim, animate, frames=e)
anim.save(path_file + '_anim.mp4', fps=10, dpi=200)
print(path_file)
