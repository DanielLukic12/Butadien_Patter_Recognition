# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:06:10 2017

@author: lukic
"""
import matplotlib.pyplot as plt


def plot_input(input_data_un, input_data, show):
    """Plotting input data normalized and unnormalized

    Args:
        input_data_un: Input data not normalized
        input_data: Input data normalized
        show: Show figure if True
    """
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(input_data_un.T[0:200])
    axarr[0].set_title('Normalized Input Data')
    axarr[0].set_ylabel('Distance/Angle in a.u.')
    axarr[0].set_xlabel('Data Points')
    axarr[0].grid()
    axarr[1].plot(input_data.T[0:200])
    axarr[1].set_title('Normalized Input Data')
    axarr[1].set_ylabel('Distance/Angle in a.u.')
    axarr[1].set_xlabel('Data Points')
    axarr[1].grid()
    if show is True:
        plt.show()
    else:
        plt.close(fig)


def plot_target(target_to_plot, show):
    """Plotting target data normalized

    Args:
        target_to_plot: Target data normalized
        show: Show figure if True
    """
    fig = plt.figure()
    plt.plot(target_to_plot.T[0:200])
    plt.ylabel('Energy Normalized in a.u.', fontsize=14)
    plt.xlabel('Data Points', fontsize=14)
    plt.title('Normalized Target Data', fontsize=16)
    plt.grid()
    if show is True:
        plt.show()
    else:
        plt.close(fig)


def plot_tested(reference_data, prediction, show, path_to_save):
    """Plotting prediction and reference data

    Args:
        reference_data: Reference data
        input_data: prediction data
        show: Show figure if True
        path_to_save: Path where to save figure
    """
    fig = plt.figure()
    plt.figure(figsize=(12, 4))
    plt.plot(reference_data, 'ro-')
    plt.plot(prediction, 'bx')
    plt.xlabel("Data Points", fontsize=14)
    plt.ylabel("Energies in Hartree", fontsize=14)
    plt.title("Predicted and Reference Energies",
              fontsize=16)
    plt.legend(['Reference Energies', 'Predicted Energies'])
    plt.savefig(path_to_save + "_pred" + ".png")
    if show is True:
        plt.show()
    else:
        plt.close(fig)


def plot_against(reference, prediction, show, path_to_save):
    """Plotting reference data against prediction data not normalized

    Args:
        reference_data: Reference data
        input_data: prediction data
        show: Show figure if True
        path_to_save: Path where to save figure
    """
    fig = plt.figure()
    plt.figure(figsize=(12, 4))
    plt.scatter(reference, prediction, marker='x', color='b')
    plt.plot(reference, reference, 'r-')
    plt.grid()
    plt.xlabel(r'$E_{ref}$ in Hartree', fontsize=14)
    plt.ylabel(r'$E_{pred}$ in Hartree', fontsize=14)
    plt.legend(['Predicted', 'Reference'])
    plt.title("Predicted and Reference Energies plotted against", fontsize=16)
    plt.savefig(path_to_save + "_predVSref" + ".png")
    if show is True:
        plt.show()
    else:
        plt.close(fig)

def plot_errors(training_error, test_error, show, path_to_save, epoch):
    """Plotting reference data against prediction data not normalized

    Args:
        training_error: List of training errors for each epoch
        test_error: List of training errors for each epoch
        show: Show figure if True
        path_to_save: Path where to save figure
        epoch: Epoch of convergende or max epoch
    """
    ax = plt.axes()
    plt.semilogy(training_error, lw=2, color='r')
    plt.semilogy(test_error, lw=2, color='b')
    plt.legend(('MSE Train', 'MSE Test'))
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title("Mean Squared Error for Training and Testing", fontsize=16)
    plt.grid()
#    plt.savefig(path_to_save + "_errors" + ".png")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Normed Energies")
    ax.set_title("Training and Testing Results for Max Epoch: " + str(epoch))
    ax.legend(['Trained Energies', 'Reference Energies'])
    if show is True:
        plt.show()
    else:
        plt.close(ax)
#    title = ax.text(4, 2.5, "")


def plot_hist(target_data, show):
    """Plotting target data normalized in a histogram

    Args:
        target_to_plot: Target data normalized
        show: Show figure if True
    """

    fig = plt.figure()
    n, bins, patches = plt.hist(target_data, bins=50,
                                histtype='bar', rwidth=0.8)
    plt.ylabel('Occurrence', fontsize=14)
    plt.xlabel('Energy Normalized in a.u.', fontsize=14)
    plt.title('Occurrence of Normalized Energies', fontsize=16)
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    if show is True:
        plt.show()
    else:
        plt.close(fig)
