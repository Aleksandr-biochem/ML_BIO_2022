import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP, FP, TN, FN = 0, 0, 0, 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                TP += 1
            elif y_true[i] == 0:
                FP += 1
        elif y_pred[i] == 0:
            if y_true[i] == 1:
                FN += 1
            elif y_true[i] == 0:
                TN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = TP / (TP + 0.5*(FP+FN))
    accuracy = (TP + TN) / len(y_pred)

    return (precision, recall, f1, accuracy)


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    
    comparison = [x == y for x,y in zip(y_pred, y_true)]
    
    return sum(comparison) / len(y_pred)


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    ybar = np.sum(y_true) / len(y_true)
    SSres = np.sum((y_true - y_pred)**2)
    SStot = np.sum((y_true - ybar)**2)
    
    return 1 - (SSres / SStot)


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    return np.sum((y_true - y_pred)**2) / len(y_true)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    return np.sum((abs(y_true - y_pred))) / len(y_true)
