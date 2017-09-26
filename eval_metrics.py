import numpy as np


def evaluate_acc_PRF(predictions, y_list, label):
    # encoding
    pred_labels = np.asarray([1 if p == label else 0 for p in predictions])
    true_labels = np.asarray([1 if y == label else 0 for y in y_list])
    # print('pred_labels.shape:', pred_labels.shape)
    # print('true_labels.shape:', true_labels.shape)
    # print('len(predictions):', len(predictions))
    # print('len(y_list):', len(y_list))
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1. 815
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.  1011
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
    precision = TP / (TP + FP)
    #print('TP'+str(TP))
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    #print(str(recall))
    #print(str(accuracy))
    print('TP + TN + FP + FN = ', TP + TN + FP + FN)
    f_score = (2 * precision * recall) / (precision + recall)
    #print(str(f_score))
    return accuracy, precision, recall, f_score