import random
import torch
import numpy as np
import copy
from configs import get_args
from scipy.fftpack import fft,ifft
args = get_args()
import math
def gausssiam_wihte(sentence):
    snr = np.random.randint(6,12)
    P_signal = torch.sum(torch.norm(sentence,p=2,dim=-1) ** 2) /len(sentence)#torch.from_numpy()
    P_noise = P_signal / 10 ** (snr / 10.0)
    return sentence + torch.randn(len(sentence)) * math.sqrt(P_noise)

def random_zero(sentence):
    time_len = len(sentence)
    mask_num = np.random.randint(15,20)
    max_mask_time = np.random.randint(1,3)
    samples =sentence.clone() # 直接把a赋给aa 其值会同时改变.clone()���ڴ��ڼ���ͼ   .detach()�����ڴ治�ڼ���ͼ   .clone().detach()���ڴ治�ڼ���ͼ
    #samples.flags.writeable = True
    for i in range(mask_num):
        t = np.random.uniform(low=1.0, high=max_mask_time+0.1)
        t = int(t)
        t0 = np.random.randint(0, time_len - t)
        samples[t0: t0 + t] = 0
    return samples

def fft_ifft(sentence):
    fft_a = torch.fft.fft(sentence-torch.mean(sentence))
    a_f = torch.fft.ifft(fft_a)
    return a_f.real#+torch.mean(sentence)

def add_noise(sentence):
    max_a,_=torch.max(sentence,dim=-1)
    min_a,_=torch.min(sentence,dim=-1)
    max_a=max_a.detach().cpu()
    min_a=min_a.detach().cpu()
    samples=torch.Tensor(len(sentence)).uniform_(min_a,max_a*0.8)
    #samples = np.random.uniform(low=min_a, high=max_a*0.8, size=len(sentence)).astype(np.float32)
    s_with_bg = sentence+samples * torch.Tensor(1).uniform_(0, 0.01)
    return s_with_bg

def text_augment(sentence):
    prob=random.random()
    if prob>0.8:
        sample= random_zero(sentence)
    elif prob>0.6 and prob<0.8 :
        sample=add_noise(sentence)
    elif prob>0.5 and prob<0.6:
        sample=fft_ifft(sentence)
    else:
        sample = gausssiam_wihte(sentence)
    return sample


def text_augmentations(sentence):
    #sentence=np.array(sentence.detach().cpu(),dtype=np.float32)

    sentence1 = text_augment(sentence)
    sentence2 = text_augment(sentence)
    #sentence1,sentence2=torch.tensor(sentence1).to('cuda', non_blocking=True),torch.tensor(sentence2).to('cuda', non_blocking=True)
    return sentence1,sentence2
    # 混合叠加

if __name__ == "__main__":
    sentence=torch.Tensor(100).uniform_(1,2)
    s1=gausssiam_wihte(sentence)
    print()