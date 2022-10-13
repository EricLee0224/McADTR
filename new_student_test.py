import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from new_student import _Teacher, TeacherOrStudent,StudentTrans
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,7'
    
    st_id = 0  # student id, start from 0.
    # image height and width should be multiples of sL1∗sL2∗sL3...
    rimH = 512
    rimW = 512

    imH = 32
    imW = 32
    patch_size = 17
    batch_size = 8
    small_batch=1024
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    multiprocess = multiPoolPrepare(patch_size, patch_size)
    work_dir = 'work_dir/'
    dataset_dir = '../../MAD/'

    device = torch.device('cuda:0')

    trans = transforms.Compose([
        transforms.Resize((rimH, rimW)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 
    dataset = datasets.ImageFolder(dataset_dir, transform=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    # 重新输出类别标签
    print(dataset.class_to_idx)
