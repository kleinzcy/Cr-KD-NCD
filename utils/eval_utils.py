# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : eval.py
# DATE : 2022/8/27 17:39
import numpy as np
# import torch
# import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
# from scipy.optimize import linear_sum_assignment as linear_assignment
from typing import List
from torch.utils.tensorboard import SummaryWriter
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pkl
from utils.dataset_info import *
import ipdb
from config import osr_split_dir


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mapping, w = compute_best_mapping(y_true, y_pred)
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size


def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


def cluster_eval(y_true, y_pred):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    y_true = y_true.astype(int) - y_true.min()

    mapping, w = compute_best_mapping(y_true, y_pred)
    novel_acc = sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size

    novel_nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    novel_ari = metrics.adjusted_rand_score(y_true, y_pred)

    results = {"novel": novel_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari}

    return results


def split_cluster_acc_v1(y_true, y_pred, num_seen):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = y_true < num_seen
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    seen_acc = np.mean(y_true[mask] == y_pred[mask])

    novel_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    novel_nmi = metrics.normalized_mutual_info_score(y_true[~mask], y_pred[~mask])
    novel_ari = metrics.adjusted_rand_score(y_true[~mask], y_pred[~mask])

    all_acc = cluster_acc(y_true, y_pred)
    all_nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    all_ari = metrics.adjusted_rand_score(y_true, y_pred)

    return {"all": all_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": seen_acc,
            "novel": novel_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari}


def split_cluster_acc_v2(y_true, y_pred, num_seen):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[y_true < num_seen])
    new_classes_gt = set(y_true[y_true >= num_seen])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    all_nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    all_ari = metrics.adjusted_rand_score(y_true, y_pred)

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])

    new_acc /= total_new_instances
    results = {"all": total_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": old_acc, "novel": new_acc}

    return results


def split_cluster_acc_v3(y_true, y_pred, num_seen):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[y_true < num_seen])
    new_classes_gt = set(y_true[y_true >= num_seen])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # only mapping unseen classes
    unseen_w = w[num_seen:, num_seen:]
    ind = linear_sum_assignment(unseen_w.max() - unseen_w)

    ind = np.vstack(ind).T
    seen_ind = np.zeros((num_seen, 2), dtype=ind.dtype)
    seen_ind[:, 0] = np.arange(num_seen)
    seen_ind[:, 1] = np.arange(num_seen)
    ind = np.concatenate((seen_ind, ind + num_seen), axis=0)
    # ind_map = {i:i for i in range(num_seen)}
    ind_map = {j: i for i, j in ind}

    total_acc = sum([w[i, j] for j, i in ind_map.items()]) * 1.0 / y_pred.size
    all_nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    all_ari = metrics.adjusted_rand_score(y_true, y_pred)

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])

    new_acc /= total_new_instances
    results = {"all": total_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": old_acc, "novel": new_acc}

    return results


EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
    'v3': split_cluster_acc_v3,
}


def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, writer: SummaryWriter=None,
                        print_output=False):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'

        if writer is not None:
            writer.add_scalars(log_name,
                               {'Old': old_acc, 'New': new_acc,
                                'All': all_acc}, T)

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return


@torch.no_grad()
def hungarian_evaluate(predictions, targets, offset=0):
    # Hungarian matching
    targets = targets - offset
    predictions = predictions - offset
    predictions_np = predictions.numpy()
    num_elems = targets.size(0)

    # only consider the valid predicts. rest are treated as misclassification
    valid_idx = np.where(predictions_np >= 0)[0]
    predictions_sel = predictions[valid_idx]
    targets_sel = targets[valid_idx]
    num_classes = torch.unique(targets).numel()
    num_classes_pred = torch.unique(predictions_sel).numel()

    match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred,
                             targets_k=num_classes)  # match is data dependent
    reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions_sel == int(pred_i)] = int(target_i)

    # Gather performance metrics
    reordered_preds = reordered_preds.numpy()
    acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(
        num_elems)  # accuracy is normalized with the total number of samples not only the valid ones
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())

    return {'acc': acc * 100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res
