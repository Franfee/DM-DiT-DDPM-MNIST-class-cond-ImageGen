import torch
from models.dit import DiT
import matplotlib.pyplot as plt
from util.diffusion import Diffusion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

noise_helper = Diffusion(T=1000, DEVICE=DEVICE)  # 扩散去噪过程描述

model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(DEVICE)  # 模型
model.load_state_dict(torch.load('result/model-42000.pth', map_location=torch.device('cpu')))

# 生成噪音图
batch_size = 10
x = torch.randn(size=(batch_size, 1, 28, 28))
y = torch.arange(start=0, end=10, dtype=torch.long)
# 逐步去噪得到原图
steps = noise_helper.backward_denoise(model, x, y)
# 绘制数量
num_imgs = 20
# 绘制还原过程
plt.figure(figsize=(15, 15))
for b in range(batch_size):
    for i in range(0, num_imgs):
        idx = int(noise_helper.T / num_imgs) * (i + 1)
        # 像素值还原到[0,1]
        final_img = (steps[idx][b].to('cpu') + 1) / 2
        # tensor转回PIL图
        final_img = final_img.permute(1, 2, 0)
        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img)
plt.savefig("result/inf.png")
plt.show()
