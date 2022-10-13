"""
Implementation of Chapt3.2 about students net training in the 'uninformed students' paper.

Author: Luyao Chen
Date: 2020.10
"""

from cgitb import small
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from new_student import _Teacher, TeacherOrStudent, StudentTrans
from fast_dense_feature_extractor import *

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    st_id = 0  # student id, start from 0.
    # image height and width should be multiples of sL1∗sL2∗sL3...
    rimH = 128
    rimW = 128

    imH = 128
    imW = 128
    # 对应teacher 一张一张学
    patch_size = 17
    batch_size = 1
    # 每张图上采样多少张
    small_batch=512
    # 1/5/10/20
    epochs = 20
    # 其余不变
    lr = 1e-4
    weight_decay = 1e-5
    # 池化trick
    multiprocess = multiPoolPrepare(patch_size, patch_size)
    work_dir = 'work_dir/'
    dataset_dir = '../data/MAD1/'

    device = torch.device('cuda')

    trans = transforms.Compose([
        transforms.Resize((rimH, rimW)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(dataset_dir, transform=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True,drop_last=True)

    #
    trans_t = transforms.Compose([
        transforms.Resize((rimH, rimW)),
        transforms.RandomCrop((imH,imW)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_t = datasets.ImageFolder(dataset_dir, transform=trans_t)
    dataloader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


    student = StudentTrans()
    student = nn.DataParallel(student).to(device)

    _teacher = _Teacher(patch_size)
    checkpoint = torch.load(work_dir + '_teacher' + str(patch_size) + '.pth', torch.device('cpu'))
    _teacher.load_state_dict(checkpoint)
    teacher = TeacherOrStudent(patch_size, _teacher, imH, imW)
    teacher = nn.DataParallel(teacher).to(device)
    teacher.eval()
    #
    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for data, _ in tqdm(dataloader):
            data = data.to(device)
            t_out = teacher(data)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)
    #
    optim = torch.optim.Adam(student.parameters(), lr=lr,
                            weight_decay=weight_decay)
    #
    iter_num = 1
    for i in range(epochs):
        for data, labels in dataloader_t:
            data = data.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # teacher_output = teacher(data)
                teacher_output = (teacher(data) - t_mu) / torch.sqrt(t_var)
            bz=data.size(0)
            # 池化加上一圈patch
            data1=multiprocess(data)
            # 逐个像素滑动窗口，得到列向量
            new_data=nn.Unfold(patch_size,1)(data1)
            #
            x = new_data.transpose(2, 1).contiguous()
            #view成不同的batch，bz原始输出尺寸，imhimw生成了多少17*17patch
            x = x.view(bz*imH*imW,3,17,17)


            # view一下T的输出方便训练
            teacher_output = teacher_output.view(bz*imH*imW, 128)
            total_loss = 0
            t = bz * imH * imW // small_batch
            for i1 in tqdm(range(t)):
                ndata = x[i1 * t: (i1*t+small_batch)]
                nteacher_output = teacher_output[i1*t:(i1 * t + small_batch)]
                nlabels = labels.repeat(small_batch)
                student_output = student(ndata,nlabels)
                loss = F.mse_loss(student_output, nteacher_output)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss+=loss.item()

            if iter_num % 1 == 0:
                print('epoch: {}, iter: {}, loss: {}'.format(i + 1, iter_num, total_loss/(bz*imH*imW//small_batch)))
                torch.save(student.state_dict(), work_dir + 'studenttrans_iter' + str(patch_size) + '_' + str(st_id) + '.pth')
            iter_num += 1
        iter_num = 0
        torch.save(student.state_dict(), work_dir + 'studenttrans' + str(patch_size) + '_' + str(st_id) + '.pth')

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    print('Saving model to work_dir...')
    torch.save(student.state_dict(), work_dir + 'studenttrans' + str(patch_size) + '_' + str(st_id) + '.pth')