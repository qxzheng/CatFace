# -*- coding: utf-8 -*-
import shutil
import os
import os.path as osp
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
# 确保目标目录存在
#weight = "/home/TransFace/work_dirs/ms1mv2_vit_l/model.pt"
#weight = "/home/TransFace/work_dirs/ms1mv2_vit_s/model.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/model.pt"
#weight_best = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/"
weight_best = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_l_catface/"

#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/5w10e_095100fft_final.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/11w10e_normal.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/11w10e_normal.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/11w10e_normal.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/5w10e_normal.pt"
weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_l_catface/11W15e_normal.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_l_catface/11W15e_050100fft.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/ms1mv2_r50_catface/model.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/ms1mv2_r50_catface/model.pt"
#weight = "/home/zqx/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s_catface/model_15e_original.pt"
#weight = "/home/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s/model_20e11w03sampleFFT095100.pt"
#weight = "/home/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s/model10ewavelet.pt"
#weight = "/home/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s/model_newlayertrainable095100.pt"
#weight = "/home/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s/model10enormaladm.pt"
#weight = "/home/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_s/model10e0707adamw.pt"
#weight = "/home/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc03_40epoch_64gpu_vit_b/model.pt"
name = "vit_s_dp005_mask_0"
#name = "vit_b_dp005_mask_005"
#name = "vit_l_dp005_mask_005"
#0weight = "/home/TransFace/work_dirs/ms1mv2_vit_s/model.pt"
#name = "vit_s_dp005_mask_0"
#name = "r50"

#from config import config as conf
#from model import FaceMobileNet
from backbones import get_model
import json

# 用于存储静态变量的文件路径
file_path = "static_var.json"

def read_static_var():
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("static_var", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

def write_static_var(value):
    with open(file_path, "w") as f:
        json.dump({"static_var": value}, f)

def increment_static_var():
    value = read_static_var()
    value += 1
    write_static_var(value)
    print(f"Static variable value: {value}")

# 测试




_static_threshold = 0
_static_accuracy = 0
#testfile1 = "filtered_cat_output.csv"
testfile1 = "filtered2w_new_pairs.csv"
test_batch_size = 512
#test_root = "/home/zqx/insightface/recognition/arcface_torch/petface_combine_before_20240802_align_112_0/"
#test_root = "/home/zqx/pet-large-model/testCode/arcface/"
test_root = "/home/zqx/data/clear_before_20241031_align_112_0/"
#test_root = ""
device = 'cuda'
def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split(',')
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i : end])
    # 打印 images 的张数
    print(f"Total image batch count: {len(res)}")
    return res


def _preprocess(images: list) -> torch.Tensor:
    res = []
    for img in images:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
        #im = Image.open(img)
        #im = transform(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        #img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)  
        res.append(img)
    data = torch.stack(res, dim=0)  # shape: (batch, 128, 128)
    #data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    #data = data[:, None, :, :]    # shape: (batch, 1, 128, 128)
    return data


def featurize(images: list, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
        #flipped_image = torch.flip(data, dims=[1])  # 水平翻转
        #result_flipped = net(flipped_image)
        #features = (features + result_flipped) / 2
        #features = result_flipped
    res = {img: feature for (img, feature) in zip(images, features)}
    return res


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def compute_accuracy(feature_dict, pair_list, test_root):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()
    # 打印 pairs 的条数
    print(f"Total pairs count: {len(pairs)}")
    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split(',')
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold

def runtest(epoch=0, weight_path = weight, name = name):
    if epoch == 0:
        _static_threshold = 0
        _static_thaccuracy = 0
        write_static_var(epoch)
    weight = weight_path+'/model.pt'
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    testfile = test_root + testfile1
    images = unique_image(testfile)
    images = [osp.join(test_root, img) for img in images]
    groups = group_image(images, test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, net, device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, testfile, test_root) 
    if read_static_var() < accuracy and epoch > 10:
        #_static_threshold = threshold
        #_static_thaccuracy = accuracy
        write_static_var(accuracy)
        # 源文件路径
        source_file = weight 
        #print(source_file)
        # 目标文件路径
        destination_file = f"{weight_path}/model{epoch}ac{accuracy:.4f}th{threshold:.4f}.pt"
        # 拷贝文件
        print(destination_file)
        shutil.copy(source_file, destination_file)
    if epoch == 0:
        print(
        #f"Test Model: {conf.test_model}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Threshold: {threshold:.4f}\n"
        )
    return accuracy, threshold

if __name__ == '__main__':

    #model = FaceMobileNet(conf.embedding_size)
    #model = nn.DataParallel(model)
    #model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    #model.eval()
    runtest(0, weight, name)
    '''
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    testfile = test_root + testfile
    images = unique_image(testfile)
    images = [osp.join(test_root, img) for img in images]
    groups = group_image(images, test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, net, device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, testfile, test_root) 

    print(
        #f"Test Model: {conf.test_model}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )
    '''
