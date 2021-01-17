import os
import numpy as np
import yaml
import torch


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def augment_config(cfg):
    cfg["train_name_file"] = os.path.join(cfg["experiment_dir"], cfg["train_name_file"])
    cfg["test_name_file"] = os.path.join(cfg["experiment_dir"], cfg["test_name_file"])
    cfg["model_path"] = os.path.join(cfg["experiment_dir"], cfg["model_path"])

    return cfg


def check_config(cfg):
    if cfg["batch_size"] > len(cfg["device_ids"]):
        print("Warning: batch_size must be no more than device count")
        print("Set new batch_size as", len(cfg["device_ids"]))
        cfg["batch_size"] = len(cfg["device_ids"])

    return cfg


def accuracy_score(label_gt, label_pred):
    accus = []
    # print(torch.max(label_gt))
    all_gt_num = label_gt.shape[0]
    label0_gt_num = torch.sum(label_gt == 0).item()
    label1_gt_num = torch.sum(label_gt == 1).item()

    label0_pred_true_num = torch.sum(label_pred[label_gt == 0] == 0).item()
    label1_pred_true_num = torch.sum(label_pred[label_gt == 1] == 1).item()
    if label0_gt_num == 0:
        label0_accu = 0
    else:
        label0_accu = label0_pred_true_num / label0_gt_num

    if label1_gt_num == 0:
        label1_accu = 0
    else:
        label1_accu = label1_pred_true_num / label1_gt_num

    all_true_num = torch.sum(label_gt == label_pred).item()
    if all_gt_num == 0:
        all_accu = 0
    else:
        all_accu = all_true_num / all_gt_num

    accus.append(all_accu)
    accus.append(label0_accu)
    accus.append(label1_accu)

    return accus






