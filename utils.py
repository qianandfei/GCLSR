'''
@Author: Gordon Lee
@Date: 2019-08-09 13:48:17
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-16 16:29:07
@Description: 
'''
import math
import torch
import torch.nn as nn
import numpy as np
import copy
from gensim.models import KeyedVectors as Vectors
from line_profiler import LineProfiler
from augmentations.eda import get_only_chars
import torch.nn.functional as F
from configs import get_args


def group_attention(query, key):#一般query为解码器的部分  k 和 v为编码器的部分
    d_k = query.size(-1) # 64=d_k
    scores = torch.matmul(query.unsqueeze(1), key.unsqueeze(1).transpose(-2, -1))/math.sqrt(d_k)
    scores=torch.sigmoid(scores)
    # p_attn = F.softmax(scores, dim = -1)
    # p_attn=p_attn.masked_fill(mask1 == 0, 1e-9)
    # p_attn=p_attn.unsqueeze(2)
    #对scores的最后一个维度执行softmax，得到的还是一个tensor,
    #(30, 8, 10, 11)
    return scores.squeeze(1)
def contrastive_att(data,model):
    data_feature=model.backbone(data)
    feature_match=model.match_model(data_feature)
    #计算attention
    d_k = data.size(-1)  # 64=d_k
    scores = torch.matmul(feature_match.unsqueeze(1), data.transpose(-2, -1)) / math.sqrt(d_k)
    return scores.transpose(-2, -1) * data


    #
    # elif self.method == 'concat':
    #     self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
    #     self.v = nn.Parameter(torch.FloatTensor(hidden_size))
    #
    # def dot_score(self, hidden, encoder_output):
    #     return torch.sum(hidden * encoder_output, dim=2)
    #
    # def general_score(self, hidden, encoder_output):
    #     energy = self.attn(encoder_output)
    #     return torch.sum(hidden * energy, dim=2)
    #
    # def concat_score(self, hidden, encoder_output):
    #     energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    #     return torch.sum(self.v * energy, dim=2)
    #
    # def forward(self, hidden, encoder_outputs):
    #     # Calculate the attention weights (energies) based on the given method
    #     if self.method == 'general':
    #         attn_energies = self.general_score(hidden, encoder_outputs)
    #     elif self.method == 'concat':
    #         attn_energies = self.concat_score(hidden, encoder_outputs)
    #     elif self.method == 'dot':
    #         attn_energies = self.dot_score(hidden, encoder_outputs)
    #
    #     # Transpose max_length and batch_size dimensions
    #     attn_energies = attn_energies.t()
    #
    #     # Return the softmax normalized probability scores (with added dimension)
    #     return F.softmax(attn_energies, dim=1).unsqueeze(1)

def data_selfattention_befor_aug(query):#一般query为解码器的部分  k 和 v为编码器的部分
# query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),
#(30, 8, 11, 64)，例如30是batch.size，即当前batch中有多少一个序列；
# 8=head.num，注意力头的个数；
# 10=目标序列中词的个数，64是每个词对应的向量表示；
# 11=源语言序列传过来的memory中，当前序列的词的个数，
# 64是每个词对应的向量表示。
# 类似于，这里假定query来自target language sequence；
# key和value都来自source language sequence.
    d_k = query.size(-1) # 64=d_k
    scores = torch.matmul(query, query.transpose(-2, -1))/math.sqrt(d_k)
    scores=torch.sum(scores,dim=1,keepdim=True)
    #p_attn = F.softmax(score, dim = -1)
    #对scores的最后一个维度执行softmax，得到的还是一个tensor,
    #(30, 8, 10, 11)
    return scores*query


def data_selfattention(query, key, value, mask):#一般query为解码器的部分  k 和 v为编码器的部分
# query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),
#(30, 8, 11, 64)，例如30是batch.size，即当前batch中有多少一个序列；
# 8=head.num，注意力头的个数；
# 10=目标序列中词的个数，64是每个词对应的向量表示；
# 11=源语言序列传过来的memory中，当前序列的词的个数，
# 64是每个词对应的向量表示。
# 类似于，这里假定query来自target language sequence；
# key和value都来自source language sequence.
    args=get_args()
    mask1=copy.deepcopy(mask)
    mask = mask.unsqueeze(2)
    # b=mask.transpose(-2, -1)
    mask = torch.matmul(mask, mask.transpose(-2, -1))
    # mask = torch.BoolTensor(mask)
    d_k = query.size(-1) # 64=d_k
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
     # 先是(30,8,10,64)和(30, 8, 64, 11)相乘，
    #（注意是最后两个维度相乘）得到(30,8,10,11)，
    #代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”。
    #然后除以sqrt(d_k)=8，防止过大的亲密度。
    #这里的scores的shape是(30, 8, 10, 11)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)#将填充的词用0覆盖
        #使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
        #然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    scores=torch.sum(scores,dim=2,keepdim=True)
    #print(scores)
    # p_attn = F.softmax(scores, dim = -1)
    # p_attn=p_attn.masked_fill(mask1 == 0, 1e-9)
    # p_attn=p_attn.unsqueeze(2)
    #对scores的最后一个维度执行softmax，得到的还是一个tensor,
    #(30, 8, 10, 11)
    return scores*value
#返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）
#value=(30,8,11,64)，得到的tensor是(30,8,10,64)，
#和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11).
#注意，这里返回p_attn主要是用来可视化显示多头注意力机制。


def compute_time(fun):
    lp = LineProfiler()
    forward = lp(fun)  # 就是把函数装饰一下
    forward()
    lp.print_stats(output_unit=1)  # 打印分
    raise ValueError('test for time')

class AverageMeter(object):
    '''
    跟踪指标的最新值,平均值,和,count
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0. #value
        self.avg = 0. #average
        self.sum = 0. #sum
        self.count = 0 #count

    def update(self, val, n=1):
        self.val = val # 当前batch的val
        self.sum += val * n # 从第一个batch到现在的累加值
        self.count += n # 累加数目加1
        self.avg = self.sum / self.count # 从第一个batch到现在的平均值


def init_embeddings(embeddings):
    '''
    使用均匀分布U(-bias, bias)来随机初始化
    
    :param embeddings: 词向量矩阵
    '''
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embedding_glove(emb_format,vectors, sentence):
    '''
    加载预训练词向量

    :param emb_file: 词向量文件路径
    :param emb_format: 词向量格式: 'glove' or 'word2vec'
    :param word_map: 词表
    :return: 词向量矩阵, 词向量维度
    '''
    assert emb_format in {'glove', 'word2vec'}

    vocab = sentence  # 所有不重复的词
    vocab = get_only_chars(vocab)
    vocab = vocab.split()
    vocab = [word for word in vocab if word is not '']
    empyty_idx = 1
    if len(vocab) == 0:
        vocab.append('sickeningly')
        empyty_idx = 0
    notinvocab = []
    # print("Loading embedding...")
    cnt = 0  # 记录读入的词数
    emb_dim=300
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    # 初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
    init_embeddings(embeddings)
    # 如果不在词表上
    word_keys=vectors.keys()
    for idx, emb_word in enumerate(vocab):
        if emb_word in word_keys:  # 若 vectors不可直接索引则可以把index2word变成set
            embedding = torch.tensor(vectors[emb_word])
            cnt += 1
            embeddings[idx] = embedding
        else:
            notinvocab.append(emb_word)
            continue
        # print("Number of words read: ", cnt)
        # print("Number of OOV: ", len(vocab) - cnt)
        # print(notinvocab)
    return embeddings,empyty_idx


def load_embedding(emb_format, vectors, data):
    '''
    加载预训练词向量
    :param emb_file: 词向量文件路径
    :param emb_format: 词向量格式: 'glove' or 'word2vec'
    :param word_map: 词表
    :return: 词向量矩阵, 词向量维度
    '''
    assert emb_format in {'glove', 'word2vec'}
    vocab = data  # 所有不重复的词
    vocab = get_only_chars(vocab)
    vocab = vocab.split()
    vocab = [word for word in vocab if word is not '']
    empyty_idx=1
    if len(vocab)==0:
        vocab.append('sickeningly')
        empyty_idx=0
    notinvocab = []
    #print("Loading embedding...")
    cnt = 0  # 记录读入的词数
    emb_dim = 300
    #idx_pad=0
    embeddings =torch.FloatTensor(len(vocab), emb_dim)#len(vocab)
    # 初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
    init_embeddings(embeddings)
    for idx,emb_word in enumerate(vocab):
        if emb_word in vectors:#若 vectors不可直接索引则可以把index2word变成set
            embedding = torch.tensor(vectors[emb_word])
            cnt += 1
            #embeddings=torch.cat((embeddings,embedding.unsqueeze(0)),0)
            embeddings[idx]=embedding
        else:
            #embeddings=torch.cat((embeddings,padding_value.unsqueeze(0)),0)
            notinvocab.append(emb_word)
    #for sent in embeddings:
        #if torch.equal(sent,padding_value):
            #idx_pad+=1
    #if idx_pad==len(vocab):
        #embeddings=torch.tensor(vectors['sickeningly']).unsqueeze(0)
        #empyty_idx=0
    # print("Number of words read: ", cnt)
    # print("Number of OOV: ", len(vocab) - cnt)
    # print(notinvocab)
    return embeddings,empyty_idx
    #return [torch.tensor(embed) for embed in embeddings if type(embed) is not 'tensor'], emb_dim
def load_embeddings(emb_file, emb_format, word_map):
    '''
    加载预训练词向量
    
    :param emb_file: 词向量文件路径
    :param emb_format: 词向量格式: 'glove' or 'word2vec'
    :param word_map: 词表
    :return: 词向量矩阵, 词向量维度
    '''
    assert emb_format in {'glove', 'word2vec'}
    
    vocab = set(word_map.keys())#所有不重复的词
    notinvocab=[]
    print("Loading embedding...")
    cnt = 0 # 记录读入的词数
    
    if emb_format == 'glove':
        
        with open(emb_file, 'r', encoding='utf-8') as f:
            emb_dim = len(f.readline().split(' ')) - 1 

        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        #初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
        init_embeddings(embeddings)
        
        
        # 读入词向量文件
        for line in open(emb_file, 'r', encoding='utf-8'):
            line = line.split(' ')
            emb_word = line[0]

            # 过滤空值并转为float型
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            # 如果不在词表上
            if emb_word not in vocab:

                continue
            else:
                cnt+=1

            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

        print("Number of words read: ", cnt)
        print("Number of OOV: ", len(vocab)-cnt)

        return embeddings, emb_dim
    
    else:
        
        vectors = Vectors.load_word2vec_format(emb_file,binary=True)
        print("Load successfully")
        emb_dim = 300
        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        #初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
        init_embeddings(embeddings)
        
        for emb_word in vocab:
            
            if emb_word in vectors:
                embedding = vectors[emb_word]
                cnt += 1
                embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
                
            else:
                notinvocab.append(emb_word)
                continue
            
        print("Number of words read: ", cnt)
        print("Number of OOV: ", len(vocab)-cnt)
        print(notinvocab)
        
        return embeddings, emb_dim

def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪防止梯度爆炸

    :param optimizer: 需要梯度裁剪的优化器
    :param grad_clip: 裁剪阈值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                # inplace操作，直接修改这个tensor，而不是返回新的
                # 将梯度限制在(-grad_clip, grad_clip)间
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy(logits, targets):
    '''
    计算单个batch的正确率
    :param logits: (batch_size, class_num)
    :param targets: (batch_size)
    :return: 
    '''
    corrects = (torch.max(logits, 1)[1].view(targets.size()).data == targets.data).sum()
    return corrects.item() * (100.0 / targets.size(0))

def adjust_learning_rate(optimizer, current_epoch):
    '''
    学习率衰减
    '''
    frac = float(current_epoch - 20) / 50
    shrink_factor = math.pow(0.5, frac)
    
    print("DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))


def save_checkpoint(model_name, data_name, epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    '''
    保存模型
    
    :param model_name: model name
    :param data_name: SST-1 or SST-2,
    :param epoch: epoch number
    :param epochs_since_improvement: 自上次提升正确率后经过的epoch数
    :param model:  model
    :param optimizer: optimizer
    :param acc: 每个epoch的验证集上的acc
    :param is_best: 该模型参数是否是目前最优的
    '''
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + data_name + '_' + model_name + '.pth'
    torch.save(state, 'checkpoints/' + filename)
    # 如果目前的checkpoint是最优的，添加备份以防被重写
    if is_best:
        torch.save(state, 'checkpoints/' + 'BEST_' + filename)


def train(train_loader, model, criterion, optimizer, epoch, vocab_size, print_freq, device, grad_clip=None):
    '''
    执行一个epoch的训练
    
    :param train_loader: DataLoader
    :param model: model
    :param criterion: 交叉熵loss
    :param optimizer:  optimizer
    :param epoch: 执行到第几个epoch
    :param vocab_size: 词表大小
    :param print_freq: 打印频率
    :param device: device
    :param grad_clip: 梯度裁剪阈值
    '''
    # 切换模式(使用dropout)
    model.train()
    
    losses = AverageMeter()  # 一个batch的平均loss
    accs = AverageMeter()  # 一个batch的平均正确率
    
    for i, (sents, labels) in enumerate(train_loader):
        
        # 移动到GPU
        sents = sents.to(device)
        targets = labels.to(device)
        
        # 前向计算
        logits = model(sents)
        
        # 计算整个batch上的平均loss
        loss = criterion(logits, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        
        # 更新参数
        optimizer.step()
        
        # 计算准确率
        accs.update(accuracy(logits, targets))
        losses.update(loss.item())
        
        # 打印状态
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          loss=losses,
                                                                          acc=accs))



def validate(val_loader, model, criterion, print_freq, device):
    '''
    执行一个epoch的验证(跑完整个验证集)

    :param val_loader: 验证集的DataLoader
    :param model: model
    :param criterion: 交叉熵loss
    :param print_freq: 打印频率
    :param device: device
    :return: accuracy
    '''
    
    #切换模式
    model = model.eval()

    losses = AverageMeter()  # 一个batch的平均loss
    accs = AverageMeter()  # 一个batch的平均正确率

    # 设置不计算梯度
    with torch.no_grad():
        # 迭代每个batch
        for i, (sents, labels) in enumerate(val_loader):

            # 移动到GPU
            sents = sents.to(device)
            targets = labels.to(device)

            # 前向计算
            logits = model(sents)

            # 计算整个batch上的平均loss
            loss = criterion(logits, targets)
            
            # 计算准确率
            accs.update(accuracy(logits, targets))
            losses.update(loss.item())

            if i % print_freq  == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader),
                                                                                loss=losses, acc=accs))
        # 计算整个验证集上的正确率
        print('LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}\n'.format(loss=losses, acc=accs))

    return accs.avg


def testing(test_loader, model, criterion, print_freq, device):
    '''
    执行测试

    :param test_loader: 测试集的DataLoader
    :param model: model
    :param criterion: 交叉熵loss
    :param print_freq: 打印频率
    :param device: device
    :return: accuracy
    '''
    
    #切换模式
    model = model.eval()

    losses = AverageMeter()  # 一个batch的平均loss
    accs = AverageMeter()  # 一个batch的平均正确率

    # 设置不计算梯度
    with torch.no_grad():
        # 迭代每个batch
        for i, (sents, labels) in enumerate(test_loader):

            # 移动到GPU
            sents = sents.to(device)
            targets = labels.to(device)

            # 前向计算
            logits = model(sents)

            # 计算整个batch上的平均loss
            loss = criterion(logits, targets)
            
            # 计算准确率
            accs.update(accuracy(logits, targets))
            losses.update(loss.item())
            
            if i % print_freq  == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(test_loader),
                                                                                loss=losses, acc=accs))

        # 计算整个测试集上的正确率
        print('LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}'.format(loss=losses, acc=accs))

    return accs.avg




