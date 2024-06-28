from torch.utils.data import DataLoader
from util.dataset import MNIST
from util.diffusion import Diffusion
import torch
from torch import nn
import os
from models.dit import DiT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

dataset = MNIST()  # 数据集
noise_helper = Diffusion(T=1000, DEVICE=DEVICE)  # 扩散去噪过程描述

model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(DEVICE)  # 模型

# 已经训练的迭代次数
iter_count = 38000
try:  # 加载模型
    model.load_state_dict(torch.load(f'result/model-{iter_count}.pth'))
except:
    pass

optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

loss_fn = nn.L1Loss()  # 损失函数(绝对值误差均值)

'''
    训练模型
'''

EPOCH = 100
BATCH_SIZE = 500

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True)  # 数据加载器
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)  # 数据加载器

model.train()


for epoch in range(EPOCH):
    for imgs, labels in dataloader:
        x = imgs * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
        t = torch.randint(0, noise_helper.T, (imgs.size(0),))  # 为每张图片生成随机t时刻
        y = labels

        x, noise = noise_helper.forward_add_noise(x, t)  # x:加噪图 noise:噪音

        pred_noise = model(x.to(DEVICE), t.to(DEVICE), y.to(DEVICE))

        loss = loss_fn(pred_noise, noise.to(DEVICE))

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if iter_count % 1000 == 0:
            print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss))
            torch.save(model.state_dict(), f'result/model-{iter_count}.pth')
            # os.replace('result/.model.pth','result/model.pth')
        iter_count += 1
