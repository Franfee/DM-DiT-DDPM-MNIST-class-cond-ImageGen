import os
import shutil


for root,dirs,files in os.walk('./'):
    root_silit=root.split('\\')
    if '__pycache__' in root_silit or '.ipynb_checkpoints' in root_silit:
        print('删除：',root)
        shutil.rmtree(root)

print('操作完成')
