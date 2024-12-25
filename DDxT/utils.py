import csv
import torch
import torch.nn.functional as F


def mean(x):
    result = sum(x) / len(x)
    return result

def evaluate_ddx(true, pred):
    """
    evaluates differential diagnosis accuracy
    :param true: ground truth sequence of labels
    :param pred: decoder sequence of predictions
    :return: accuracy
    """
    mask = torch.where(true > 0, 1., 0.)
    pred_sum = pred.sum(dim=-1)
    pred = torch.argmax(pred, dim=-1)

    acc = (true == pred).float() * mask
    acc = torch.sum(acc) / torch.sum(mask)
    return acc, pred_sum, pred


def evaluate_cls(true, pred):
    """
    evaluates accuracy of pathology classification
    :param true: ground truth labels of pathology
    :param pred: predicted one-hot approximation of classifier
    :return:
    """
    probas = F.softmax(pred, dim=1)
    pred_proba = torch.max(probas, dim=-1)[0]
    pred = torch.argmax(pred, dim=-1)
    acc = (true == pred).float().mean()
    return acc, pred_proba


def save_history(file, history, mode='w'):
    """
    writes history to a csv file
    :param file: name of the file
    :param history: list of history
    :param mode: writing mode
    :return: None
    """
    with open(file, mode) as f:
        writer = csv.writer(f)
        history = [line.replace(':', ',').split(',') for line in history]
        [writer.writerow(line) for line in history]

def preprocess_ddx_f1(true, pred):
    mask = torch.where(true > 0, 1., 0.)
    pred = torch.argmax(pred, dim=-1)
    true_labels = torch.masked_select(true, mask.bool()).cpu().tolist()
    pred_labels = torch.masked_select(pred, mask.bool()).cpu().tolist()
    return true_labels, pred_labels