from gensim.models import word2vec
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import numpy as np
from augmentations.text_aug import text_augmentations
from gensim.models import KeyedVectors
from nltk import word_tokenize
#import torch.utils.data as Data
from nltk.corpus import stopwords
import pandas
from tqdm import tqdm
#import torch
import argparse
import yaml

import nltk
import  re
import logging



# def load_word2vec(vocab, path="../GoogleNews-vectors-negative300.bin"):
#     print("loading word2vec...")
#     word2vectors = KeyedVectors.load_word2vec_format(path, binary=True)
#     wv_matrix = []#[]为列表   wv_matrix=np.empty(1)只有列表才能使用append方法,list只支持len()
#     wv_matrixs = []
#     word_dim = word2vectors.vector_size
#     for word in vocab:
#         for words in word:
#             if words in word2vectors:
#                 wv_matrix.append(word2vectors.word_vec(words))
#             else:
#                 wv_matrix.append(np.random.uniform(-0.01, 0.01, word_dim).astype('float32'))
#
#         wv_matrixs.append(wv_matrix)
#         wv_matrix=[]
#     # wv_matrix.append(np.random.uniform(-0.01, 0.01, word_dim).astype('float32'))
#     # wv_matrix.append(np.zeros(word_dim).astype('float32'))
#     wv_matrixs = np.array(wv_matrixs,dtype=object)#转换成数组也需要所有的元素长度一致若不一致需加上object
#     return wv_matrixs
#传统word2vec训练
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)#用来记录日志
# sentences = word2vec.Text8Corpus('text8')
# model = word2vec.Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
# model.save('text8.model')
# print(model['man'])

model=word2vec.Word2Vec.load('text8.model')
a=model['beautiful']
b=model['voluptuous']
c=model['exquisite']
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(a,'-or')
# # ax.plot(aa,'-^k')#o,*,x,+,v,<,>,s,d
# ax.plot(b,'-*b')
# # ax.hist(aaa)
# plt.show()
#给信号添加高斯白噪声
# snr=6
# P_signal = np.sum(abs(a)**2)/len(a)
# P_noise = P_signal/10**(snr/10.0)
# aa=a+np.random.randn(len(a)) * np.sqrt(P_noise)
# aaa=np.random.randn(len(a)) * np.sqrt(P_noise)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(a,'-or')
# ax.plot(aa,'-^k')#o,*,x,+,v,<,>,s,d
# ax.plot(b,'-*b')
# # ax.hist(aaa)
# plt.show()
#置零平移
# max_ratio=0.05
# samples = a.copy()
# max_shifts = len(samples) * max_ratio  # around 5% shift
# shifts_num = np.random.randint(-max_shifts, max_shifts)
# if shifts_num==0:
#    shifts_num=-3
# print(shifts_num)
# if shifts_num > 0:
#     # time advance
#     temp = samples[:shifts_num]
#     samples[:-shifts_num] = samples[shifts_num:]
#     # samples[-shifts_num:] = 0
#     samples[-shifts_num:] = 0
# elif shifts_num < 0:
#     # time delay
#     temp = samples[shifts_num:]
#     samples[-shifts_num:] = samples[:shifts_num]
#     # samples[:-shifts_num] = 0
#     samples[:-shifts_num] = 0
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(a, '-or')
# ax.plot(samples, '->b')
# plt.show()
#循环平移
# max_ratio=0.05
# samples = a  # frombuffer()导致数据不可更改因此使用拷贝
# frame_num = len(samples)
# max_shifts = frame_num * max_ratio  # around 5% shift
# nb_shifts = np.random.randint(-max_shifts, max_shifts)
# samples = np.roll(samples, nb_shifts, axis=0)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(a, '-or')
# ax.plot(samples, '->b')
# plt.show()
#混合叠加




#随机置零
# time_len = len(a)
# mask_num=10
# max_mask_time=5
# aa=a.copy()#直接把a赋给aa 其值会同时改变
# aa.flags.writeable = True
# for i in range(mask_num):
#     t = np.random.uniform(low=0.0, high=max_mask_time)
#     t = int(t)
#     t0 = np.random.randint(0, time_len - t)
#     aa[t0: t0 + t]=0
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(a, '-or')
# ax.plot(aa, '->b')
# plt.show()
#傅里叶变换
fft_a=np.abs(fft(a-np.mean(a)))
fft_b=np.abs(fft(b-np.mean(b)))
a_f=ifft(fft(a-np.mean(a)))
print(a_f.real+np.mean(a))
print(a)
y2 = model.most_similar("beautiful", topn=20)
print(y2)
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(a,'-or')
abs_a_f=np.abs(a_f)
print(a_f[:1])
ax.plot(a_f,'-*b')
plt.show()
# ax.set_title('BMI wise Sum of Sales')
# ax.plot(a,color='r')
# ax.plot(b,color='k')
# ax.plot(c,color='b')
# plt.show()
# fig = plt.figure(2)
# ax = fig.add_subplot(111)
# ax.set_xlabel('BMI')
# ax.set_ylabel('Sum of Sales')
# ax.set_title('BMI wise Sum of Sales')
# ax.plot(fft_a,color='r')


# fig = plt.figure(3)
# ax = fig.add_subplot(111)
# ax.set_xlabel('BMI')
# ax.set_ylabel('Sum of Sales')
# ax.set_title('BMI wise Sum of Sales')
# ax.plot(fft_b,color='r')
# plt.show()



# print(model.most_similar('beautiful',topn=10))


# a=b=0
# with open ('text8','r',encoding='utf-8') as file:
#      line = file.read()#file.readline
#      for char in line:
#          b=b+1
#          print(char,end='')
#          if b-a ==10:
#              a=b
#              print('\n')
#          if a==100:
# #              break
# if __name__ == "__main__":
#     corpus=pandas.read_pickle('../MR.pkl')
#     sentence=np.array(corpus.sentence)
#     vocabs_vector=[]
#     vocabs=[]
#     feature_bank=[]
#     feature_labels = []
#
#     for sentences in sentence:
#         sentences=sentences.replace("'s", " is")
#         sentences = sentences.replace("'", "")
#         sentences = sentences.replace('-', " ")
#         vocab = word_tokenize(sentences)
#         # vocab=re.split('[, . ? ! - ]', sentence.strip())
#         symbol = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
#         #stops=set(stopwords.words('english'))
#         vocab = [word for word in vocab if word not in symbol] # 去除标点符号
#         #vocab = [word for word in vocab if word not in stops]#  去停用词--是否有必要？可以试验
#
#         vocabs.append(vocab)
#
#         # a=load_word2vec(vocab)
#         # print(a)
#
#     label =np.array(corpus.label,dtype=np.int8)
#     print(len(label))
#     label =torch.Tensor(label)#
#     print(label)
#     vocabs_vector=load_word2vec(vocabs)
#     print(vocabs_vector.shape)
#     # label=np.asarray(label)#np.asarray(label)
#     # vocabs_vector = torch.Tensor()#转换成tensor需各个元素长度一致
#
#     tensordata=Data.TensorDataset(vocabs_vector,label)
#     dataloader=Data.DataLoader(dataset=tensordata,batch_size=12,shuffle=False,num_workers=2)
#     for data, target in tqdm(dataloader, desc='Feature extracting', leave=False, disable=False):
#         feature_bank.append(data)
#         feature_labels.append(target)
#     # [D, N]
#     print(feature_bank.shape,feature_labels.shape)


    # a=load_word2vec(vocabs[1])
    # print(vocabs[1],'\n')
    # print(a, '\n',len(a))



