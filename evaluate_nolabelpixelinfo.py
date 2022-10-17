"""
Evaluation for mvtec_ad dataset.
Reference from https://github.com/denguir/student-teacher-anomaly-detection.

Author: Luyao Chen
Date: 2022.09
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
from new_student import _Teacher, TeacherOrStudent,StudentTrans
from mvtec_dataset import MVTec_AD
from fast_dense_feature_extractor import *


def error(student_outputs, teacher_output):
    # n*imH*imW*d
    # s_mean = 0
    # for s_out in student_outputs:
    #     s_mean += s_out
    # s_mean /= len(student_outputs)
    s_mean = torch.mean(student_outputs, dim=1)
    return torch.norm(s_mean - teacher_output, dim=3)


def variance(student_outputs):
    # s_sum = 0
    # for s_out in student_outputs:
    #     s_sum += s_out
    # s_mean = s_sum / len(student_outputs)

    # v = 0
    # for s_out in student_outputs:
    #     v += torch.norm(s_out - s_mean, dim=3)
    # v /= len(student_outputs)
    sse = torch.sum(student_outputs ** 2, dim=4)

    msse = torch.mean(sse, dim=1)
    s_mean = torch.mean(student_outputs, dim=1)
    var = msse - torch.sum(s_mean**2, dim=3)

    return var


def increment_mean_and_var(mu_N, var_N, N, batch):
    '''Increment value of mean and variance based on
       current mean, var and new batch
    '''
    # batch: (batch, h, w, vector)
    B = batch.size()[0]  # batch size
    # we want a descriptor vector -> mean over batch and pixels
    mu_B = torch.mean(batch, dim=[0, 1, 2])
    S_B = B * torch.var(batch, dim=[0, 1, 2], unbiased=False)
    S_N = N * var_N
    mu_NB = N / (N + B) * mu_N + B / (N + B) * mu_B
    S_NB = S_N + S_B + B * mu_B**2 + N * mu_N**2 - (N + B) * mu_NB**2
    var_NB = S_NB / (N + B)
    return mu_NB, var_NB, N + B


if __name__ == "__main__":
    import os
    from collections import Counter
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # add more size for multi-scale segmentation
    # 暂且只用一个学生一个教师
    patch_sizes = [17]   #17,33,65
    # num of studetns per teacher
    num_students = 1   #1
    # image height and width should be multiples of sL1∗sL2∗sL3...
    imH = 256
    imW = 256
    batch_size = 1
    print(os.getcwd())
    small_batch = 512
    work_dir = 'work_dir/'
    # class_dir = 'cable/'
    class_dir1 = 'carpet/'
    # /home/DISCOVER_summer2022/liwz/data/MVTec/bottle 
    # 0：bottle etc.设成none也成
    labels = torch.ones((1))*3
    train_dataset_dir = '~/data/MVTec/' + class_dir1 + 'train/'
    test_dataset_dir = '~/data/MVTec/' + class_dir1
    device = torch.device('cuda')

    N_scale = len(patch_sizes)

    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    trans = transforms.Compose([
        # transforms.RandomCrop((imH, imW)),
        transforms.Resize((imH, imW)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mask_trans = transforms.Compose([
        #transforms.Resize((imH, imW)),
        #transforms.CenterCrop((32, 32)),
        transforms.Resize((imH, imW), Image.NEAREST),
        transforms.ToTensor(),
    ])
    anomaly_free_dataset = datasets.ImageFolder(
        train_dataset_dir, transform=trans)
    af_dataloader = DataLoader(anomaly_free_dataset, batch_size=batch_size)
    test_dataset = MVTec_AD(test_dataset_dir, transform=trans,
                            mask_transform=mask_trans, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # padding
    multiprocess = multiPoolPrepare(17, 17)

    teachers = []
    students = []
for patch_size in patch_sizes:
    _teacher = _Teacher(patch_size)
    checkpoint = torch.load(work_dir + '_teacher' +
                            str(patch_size) + '.pth', torch.device('cpu'))
    _teacher.load_state_dict(checkpoint)
    teacher = TeacherOrStudent(patch_size, _teacher, imH, imW).to(device)
    teacher.eval()
    teachers.append(teacher)

    s_t = []
    for i in range(num_students):
        _teacher = _Teacher(patch_size)
        # 
        student = StudentTrans()
        student = nn.DataParallel(student).to(device)
        # 
        checkpoint = torch.load(work_dir + '/studenttrans' +
                                str(patch_size) + '_' + str(i) +
                                '.pth', torch.device('cpu'))
        student.load_state_dict(checkpoint)
        student.eval()
        s_t.append(student)
    students.append(s_t)
    
    for data, _ in tqdm(af_dataloader):
        data = data.to(device)
        # 
        labels = labels.to(device).long()
        for i in range(N_scale):
            for j in range(num_students):
        # 
                data1 = multiprocess(data)
                new_data = nn.Unfold(17, 1)(data1)
                bz = data.size(0)
                x = new_data.transpose(2, 1).contiguous()
                x = x.view(bz*imH * imW, 3, 17, 17)
                output=torch.zeros(bz*imH * imW,128).to(device)
                labels1=[]
                label=torch.zeros(bz*imH * imW)
                for i1 in range(bz*imH*imW//small_batch):
                    ndata = x[i1*small_batch:(i1*small_batch+small_batch)]
                    with torch.no_grad():
                        # 不加标签测试
                        a, class_1 = students[i][j](ndata,None)
                        print(class_1)
                        label[i1*small_batch:(i1*small_batch+small_batch)] = class_1
            
                print(torch.mode(label,0).values)
