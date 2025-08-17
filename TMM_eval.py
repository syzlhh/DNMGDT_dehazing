import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import  imageio
from glob import  glob
import cv2

def eval_ohaze():
    imgs_dehaze = glob('C:\\Users\Admin\Desktop\DNMGDT-main\DNMGDT-main\ohaze\\*.jpg')
    imgs_gt = 'D:\Data\O-HAZE\Data\\all\GT\\'
    ssim_dict = []
    psnr_dict = []
    for dehaze in imgs_dehaze:
        li_split = dehaze.split('\\')[-1].split('.')[0]
        gt = imgs_gt + li_split.replace('hazy', 'GT') + '.jpg'
        img_gt = cv2.imread(gt)
        img_gt = cv2.resize(img_gt, (512, 512)).astype(float) / 255.0
        img_dehaze = cv2.imread(dehaze)
        img_dehaze = cv2.resize(img_dehaze,(512, 512)).astype(float) / 255.0
        psnr0 = psnr(img_gt,img_dehaze,data_range=1.0)
        ssim0 = ssim(img_gt,img_dehaze,multichannel=True)
        print(dehaze.split('\\')[-1],"psnr:", psnr0, "ssim:", ssim0)
        ssim_dict.append(ssim0)
        psnr_dict.append(psnr0)
    print("mean ssim is ", np.mean(ssim_dict, 0))
    print("mean psnr is ", np.mean(psnr_dict, 0))
if __name__=='__main__':
    eval_ohaze()