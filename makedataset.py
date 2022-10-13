import os
import shutil
from tqdm import tqdm

path = '/home/DISCOVER_summer2022/liwz/data/MVTec'
listfile = os.listdir(path)
for i in listfile:
    kpath = '/home/DISCOVER_summer2022/liwz/data/MVTec/'+ i +'/train/good/'
    imglist = os.listdir(kpath)
    for img in tqdm(imglist):
        if not os.path.exists('/home/DISCOVER_summer2022/liwz/data/MAD1/'+i):
            os.makedirs('/home/DISCOVER_summer2022/liwz/data/MAD1/'+i)
        src = kpath + img
        tgt = '/home/DISCOVER_summer2022/liwz/data/MAD1/'+ i +'/'+img
        shutil.copyfile(src,tgt)