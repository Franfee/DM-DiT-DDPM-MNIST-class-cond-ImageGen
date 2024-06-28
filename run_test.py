

import torch
from util.diffusion import Diffusion
noise_helper = Diffusion(T=1000) # 扩散去噪过程描述

def test_dataset():
    import matplotlib.pyplot as plt 
    from util.dataset import MNIST

    ds=MNIST()
    img,label=ds[0]
    print(label)
    plt.imshow(img.permute(1,2,0))
    plt.show()


def test_diffusion():
    import matplotlib.pyplot as plt 
    from util.dataset import MNIST

    dataset=MNIST()
    
    # 原图
    x=torch.stack((dataset[0][0],dataset[1][0]),dim=0) # 2个图片拼batch, (2,1,48,48)

    # 随机时间步
    t=torch.randint(0,noise_helper.T,size=(x.size(0),))
    print('in test_diffusion t:',t)
    
    # 加噪
    x_noise=x*2-1 # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    x_noise,noise=noise_helper.forward_add_noise(x_noise,t)
    print('in test_diffusion x:',x_noise.size())
    print('test_diffusion noise:',noise.size())

    # 原图
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(x[0].permute(1,2,0))
    plt.subplot(2,2,2)
    plt.imshow(x[1].permute(1,2,0))

    # 加噪图
    plt.subplot(2,2,3)
    plt.imshow(((x_noise[0]+1)/2).permute(1,2,0))   
    plt.subplot(2,2,4)
    plt.imshow(((x_noise[1]+1)/2).permute(1,2,0))
    plt.show()


def test_time_emb():
    from models.time_emb import TimeEmbedding
    time_emb=TimeEmbedding(16)
    t=torch.randint(0,noise_helper.T,(2,))   # 随机2个图片的t时间步
    embs=time_emb(t) 
    print(embs)


def test_dit():
    from models.dit import DiT
    from models.dit_block import DiTBlock

    dit_block=DiTBlock(emb_size=16,nhead=4)
    
    x=torch.rand((5,49,16))
    cond=torch.rand((5,16))
    
    outputs=dit_block(x,cond)
    print(outputs.shape)

    # ------------------

    dit=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4)
    x=torch.rand(5,1,28,28)
    t=torch.randint(0,noise_helper.T,(5,))
    y=torch.randint(0,10,(5,))
    outputs=dit(x,t,y)
    print(outputs.shape)


if __name__=='__main__':
    test_dataset()
    test_diffusion()
    test_dit()