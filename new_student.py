import torch
from transformer import Transformer
from pencoder import NestedTensor, nested_tensor_from_tensor_list, PositionEmbeddingLearned
import torch.nn as nn
import numpy as np
from torch import nn
from fast_dense_feature_extractor import *

class _Teacher17(nn.Module):
    """
    T^ net for patch size 17.
    """

    def __init__(self):
        super(_Teacher17, self).__init__()
        self.net = nn.Sequential(
            # Input n*3*17*17
            # ???? kernel_size=5????
            nn.Conv2d(3, 128, kernel_size=6, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*12*12
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*8*8
            nn.Conv2d(256, 256, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*4*4
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            # n*128*1*1
        )
        self.decode = nn.Linear(128, 512)
        # nn.Sequential(
        # # nn.LeakyReLU(5e-3),
        # # # n*128*1*1
        # # nn.Conv2d(128, 512, kernel_size=1, stride=1),
        # # output n*512*1*1
        # )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 128)
        x = self.decode(x)
        return x


class _Teacher33(nn.Module):
    """
    T^ net for patch size 33.
    """

    def __init__(self):
        super(_Teacher33, self).__init__()
        self.net = nn.Sequential(
            # Input n*3*33*33
            nn.Conv2d(3, 128, kernel_size=3, stride=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(5e-3),
            # n*128*29*29
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*128*14*14
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(5e-3),
            # n*256*10*10
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*256*5*5
            nn.Conv2d(256, 256, kernel_size=2, stride=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(5e-3),
            # n*256*4*4
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            # n*128*1*1
        )
        self.decode = nn.Linear(128, 512)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 128)
        x = self.decode(x)
        return x


class _Teacher65(nn.Module):
    """
    T^ net for patch size 65.
    """

    def __init__(self):
        super(_Teacher65, self).__init__()
        self.net = nn.Sequential(
            # Input n*3*65*65
            nn.Conv2d(3, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*61*61
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*128*30*30
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*26*26
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*128*13*13
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*9*9
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*256*4*4
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*1*1
            # ???? kernel_size=3????
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            # n*128*1*1
        )
        self.decode = nn.Linear(128, 512)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 128)
        x = self.decode(x)
        return x


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""
    def __init__(self, decoder_dim, output_dim, use_prior=False):
        super().__init__()
        ch = 256
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))
        return self.fc_o(x)


class Teacher17(nn.Module):
    """
    Teacher network with patch size 17.
    It has same architecture as T^17 because with no striding or pooling layers.
    """

    def __init__(self, base_net: _Teacher17):
        super(Teacher17, self).__init__()
        self.multiPoolPrepare = multiPoolPrepare(17, 17)
        self.net = base_net.net

    def forward(self, x):
        x = self.multiPoolPrepare(x)
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        return x

# class Student17(nn.Module):
#     def __init__(self, ):
#         super(Student17, self).__init__()
#         self.multiprocess=multiPoolPrepare(17,17)
#         self.unfold = nn.Unfold(17,1)
#         self.input_proj = nn.Conv2d(3, 128, kernel_size=1)
#         self.query_embed = nn.Embedding(15, 128)
#         self.position_embedding = PositionEmbeddingLearned(64)
#         self.log_softmax = nn.LogSoftmax(dim=1)
#
#         self.scene_embed = nn.Linear(128, 1)
#         self.regressor_head_t = nn.Sequential(*[PoseRegressor(128, 128) for _ in range(15)])
#         self.transformer=Transformer()
#
#
#     def forward(self, x,label):
#         x = self.multiprocess(x)
#         b=x.size(0)
#         x = self.unfold(x)
#         x = x.transpose(2,1)
#         x = x.view(b,512*512,3,17,17).contiguous()
#         batchsize=128
#         out=[]
#         for i in range(512*512//batchsize):
#             xseg = x[:,i*batchsize:i*batchsize+batchsize]
#             xsegnew=xseg.view(b*batchsize,3,17,17)
#             xsamples = nested_tensor_from_tensor_list(xsegnew)
#             x1 = xsamples.tensors
#             mask = xsamples.mask
#             x_proj = self.input_proj(x1)
#             pos =self.position_embedding(x_proj)
#             local_descs = self.transformer(x_proj, mask, self.query_embed.weight, pos)[0][0]
#             scene_log_distr = self.log_softmax(self.scene_embed(local_descs)).squeeze(2)
#             _, max_indices = scene_log_distr.max(dim=1)
#             w = local_descs * 0
#             w[range(batchsize*b), max_indices, :] = 1
#             global_desc_t = torch.sum(w * local_descs, dim=1)
#             if label is not None:
#                 max_indices = label.repeat(b*batchsize)
#
#             expected_pose = torch.zeros((batchsize*b, 128)).to(global_desc_t.device).to(global_desc_t.dtype)
#             for i1 in range(batchsize*b):
#                 x_t = self.regressor_head_t[max_indices[i1]](global_desc_t[i1].unsqueeze(0))
#                 expected_pose[i, :] = x_t
#         return x

# 框架功能基本参照pose
# 对17*17/33*33/65*65的patch进行操作
class StudentTrans(nn.Module):
    def __init__(self, ):
        super(StudentTrans, self).__init__()
        # 3转128维度
        self.input_proj = nn.Conv2d(3, 128, kernel_size=1)
        # 15类-15个learnable query
        self.query_embed = nn.Embedding(15, 128)
        # 64：编码维度
        self.position_embedding = PositionEmbeddingLearned(64)
        # 类别选择log_softmax
        self.log_softmax = nn.LogSoftmax(dim=1) # 维度1上元素相加＝1
        # 将每个像素的[128维描述向量]变成[1维类别向量]备选
        self.scene_embed = nn.Linear(128, 1)
        # 构建单独15个回归器 （多层fc—）
        self.regressor_head_t = nn.Sequential(*[PoseRegressor(128, 128) for _ in range(15)])
        self.transformer = Transformer()

    def forward(self, x, label=None):
        b = x.size(0)
        # nested tensor mask
        xsamples = nested_tensor_from_tensor_list(x)
        x1 = xsamples.tensors
        mask = xsamples.mask
        # 3->128
        x_proj = self.input_proj(x1)
        # +positional Embedding
        pos = self.position_embedding(x_proj)
        # transformer输入:(x_proj, mask, self.query_embed.weight, pos)
        local_descs = self.transformer(x_proj, mask, self.query_embed.weight, pos)[0][0]
        # local_descs局部描述输出即为:[1, 15, 128]
        out = self.scene_embed(local_descs)
        scene_log_distr = self.log_softmax(out).squeeze(2) #去掉多余维度只要第3维信息
        # return最大值类别索引序号
        _, max_indices = scene_log_distr.max(dim=1)
       
        # train------------------------------------------------------------------------------
        #权重向量 选出对应类输出 没用的置0
        w = local_descs * 0
        w[range(b), max_indices, :] = 1
        # 全局描述
        global_desc_t = torch.sum(w * local_descs, dim=1)
        # 标签，训练加，测试可加可不加。
        if label is not None:
            max_indices = label
        # 输出期望类别的128维向量（pixel info）
        expected_vec = torch.zeros((b, 128)).to(global_desc_t.device)
        for i1 in range(b):
            x_t = self.regressor_head_t[max_indices[i1]](global_desc_t[i1].unsqueeze(0))
            expected_vec[i1, :] = x_t
        return expected_vec,out
        # ------------------------------------------------------------------------------

class Teacher33(nn.Module):
    """
    Teacher network with patch size 33.
    """

    def __init__(self, base_net: _Teacher33, imH, imW):
        super(Teacher33, self).__init__()
        self.imH = imH
        self.imW = imW
        self.sL1 = 2
        self.sL2 = 2
        # image height and width should be multiples of sL1∗sL2∗sL3...
        # self.imW = int(np.ceil(imW / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        # self.imH = int(np.ceil(imH / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        assert imH % (self.sL1 * self.sL2) == 0, \
            "image height should be multiples of (sL1∗sL2) which is " + \
            str(self.sL1 * self.sL2)
        assert imW % (self.sL1 * self.sL2) == 0, \
            "image width should be multiples of (sL1∗sL2) which is " + \
            str(self.sL1 * self.sL2)

        self.outChans = base_net.net[-1].out_channels
        self.net = nn.Sequential(
            multiPoolPrepare(33, 33),
            base_net.net[0],
            base_net.net[1],
            multiMaxPooling(self.sL1, self.sL1, self.sL1, self.sL1),
            base_net.net[3],
            base_net.net[4],
            multiMaxPooling(self.sL2, self.sL2, self.sL2, self.sL2),
            base_net.net[6],
            base_net.net[7],
            base_net.net[8],
            unwrapPrepare(),
            unwrapPool(self.outChans, imH / (self.sL1 * self.sL2),
                       imW / (self.sL1 * self.sL2), self.sL2, self.sL2),
            unwrapPool(self.outChans, imH / self.sL1,
                       imW / self.sL1, self.sL1, self.sL1),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], self.imH, self.imW, -1)
        x = x.permute(3, 1, 2, 0)
        return x


class Teacher65(nn.Module):
    """
    Teacher network with patch size 65.
    """

    def __init__(self, base_net: _Teacher65, imH, imW):
        super(Teacher65, self).__init__()
        self.imH = imH
        self.imW = imW
        self.sL1 = 2
        self.sL2 = 2
        self.sL3 = 2
        # image height and width should be multiples of sL1∗sL2∗sL3...
        # self.imW = int(np.ceil(imW / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        # self.imH = int(np.ceil(imH / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        assert imH % (self.sL1 * self.sL2 * self.sL3) == 0, \
            'image height should be multiples of (sL1∗sL2*sL3) which is ' + \
            str(self.sL1 * self.sL2 * self.sL3) + '.'
        assert imW % (self.sL1 * self.sL2 * self.sL3) == 0, \
            'image width should be multiples of (sL1∗sL2*sL3) which is ' + \
            str(self.sL1 * self.sL2 * self.sL3) + '.'

        self.outChans = base_net.net[-1].out_channels
        self.net = nn.Sequential(
            multiPoolPrepare(65, 65),
            base_net.net[0],
            base_net.net[1],
            multiMaxPooling(self.sL1, self.sL1, self.sL1, self.sL1),
            base_net.net[3],
            base_net.net[4],
            multiMaxPooling(self.sL2, self.sL2, self.sL2, self.sL2),
            base_net.net[6],
            base_net.net[7],
            multiMaxPooling(self.sL3, self.sL3, self.sL3, self.sL3),
            base_net.net[9],
            base_net.net[10],
            base_net.net[11],
            unwrapPrepare(),
            unwrapPool(self.outChans, imH / (self.sL1 * self.sL2 * self.sL3),
                       imW / (self.sL1 * self.sL2 * self.sL3), self.sL3, self.sL3),
            unwrapPool(self.outChans, imH / (self.sL1 * self.sL2),
                       imW / (self.sL1 * self.sL2), self.sL2, self.sL2),
            unwrapPool(self.outChans, imH / self.sL1,
                       imW / self.sL1, self.sL1, self.sL1),
        )

    def forward(self, x):
        x = self.net(x)
        # print(x.shape)
        x = x.view(x.shape[0], self.imH, self.imW, -1)
        x = x.permute(3,1,2,0)
        return x



def _Teacher(patch_size):
    if patch_size == 17:
        return _Teacher17()
    if patch_size == 33:
        return _Teacher33()
    if patch_size == 65:
        return _Teacher65()
    else:
        print('No implementation of net wiht patch_size: ' + str(patch_size))
        return None

# 加载teacher
def TeacherOrStudent(patch_size, base_net, imH=None, imW=None):
    if patch_size == 17:
        return Teacher17(base_net)
    if patch_size == 33:
        if imH is None or imW is None:
            print('imH and imW are necessary.')
            return None
        return Teacher33(base_net, imH, imW)
    if patch_size == 65:
        if imH is None or imW is None:
            print('imH and imW are necessary.')
            return None
        return Teacher65(base_net, imH, imW)
    else:
        print('No implementation of net wiht patch_size: '+str(patch_size))
        return None

if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    net = StudentTrans()
    net = nn.DataParallel(net).cuda()
    imH = 17
    imW = 17

    batch_size=2048
    #
    x = torch.ones((batch_size, 3, imH, imW)).cuda()
    y=torch.ones((batch_size)).long().cuda()

    out=net(x,y)
    print(out.shape)
