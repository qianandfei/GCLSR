from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from transformers import BertTokenizer,BertModel,BertForMaskedLM
from configs import *
import torch.nn as nn
from models import TextCNN
from line_profiler import LineProfiler
import torch
from configs import get_args



def textcnn():
    model_opt = TextCNN.ModelConfig()
    model =TextCNN.ModelCNN(
                     kernel_num=model_opt.kernel_num,
                     kernel_sizes=model_opt.kernel_sizes,
                        model_dim=model_opt.model_dim
                             )
    return model

import os
# def bert():
#     set_deterministic(1)
#     UNCASED = './bert-base-uncased'
#     VOCAB = 'vocab.txt'
#     tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
#     bert = BertModel.from_pretrained(UNCASED)

# class Config(object):
#
#     """配置参数"""
#     def __init__(self, dataset):
#         self.model_name = 'bert'
#         self.train_path = dataset + '/data/train.txt'                                # 训练集
#         self.dev_path = dataset + '/data/valid.txt'                                    # 验证集
#         self.test_path = dataset + '/data/test.txt'                                  # 测试集
#         self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
#
#         self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = 2                                            # 类别数
#         self.num_epochs = 10                                             # epoch数
#         self.batch_size = 32                                           # mini-batch大小
#         self.pad_size = 300                                              # 每句话处理成的长度(短填长切)
#         self.learning_rate = 5e-5                                       # 学习率
#         self.bert_path = './bert_pretrain'
#         self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
#         self.hidden_size = 768




class Bert(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(args.bert_hidden_size, args.num_classes)

    def forward(self, x, attention_masks):#用attention mask是一个只有 0 和 1 组成的数组，标记哪些 tokens 是填充的，哪些不是的。掩码会告诉 BERT 中的 “Self-Attention” 机制不去处理这些填充的符号
        # context = x[0]  # 输入的句子
        # mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        last_hidden_state ,pooler_output = self.bert(x,attention_mask=attention_masks,return_dict=False,output_attentions=False,output_hidden_states=False)#返回的是encoded层和对最后一层pool（全连接）的结果，pool层的返回是特殊字符[cls]的表示向量可看做整句表示
       # _, pooled = self.bert(x, attention_mask=mask, output_hidden_states=False)
        # out = self.fc(pooled)

        return last_hidden_state



# def bert_textcnn():
# def bert_lstm():
# def word2vec_textcnn():
# def worc3vec_lstm():




def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet50w2(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


# def resnet50w4(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


# def resnet50w5(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3])
if __name__ == "__main__":
    PAD, CLS = '[PAD]', '[CLS]'
    args = get_args()
    tokenizer=BertTokenizer.from_pretrained(args.bert_path,do_lower_case=True)
    sentences=["The more pictures of him that appear in the news, the more embarrassed John becomes"]
    # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
    # max_len = 0
    # for sent in sentences:#计算文档中句子的最大长度
    #
    #     input_ids = tokenizer.encode(sentences, add_special_tokens=True)#完成的是tokenize和convert_tokens_to_ids两个动作
    #     max_len = max(max_len, len(input_ids))



   #在以输入为两个句子的任务中（例如：句子 A 中的问题的答案是否可以在句子 B 中找到）每个句子的结尾，需要添加特殊的 [SEP] 符号，该符号为这两个句子的分隔符
     #BERT 要求我们：
        # 1在句子的句首和句尾添加特殊的符号（在句子结尾添加[SEP]符号，在分类任务中需要将[cls]符号插入到每个句子的开头，填充句子要使用[PAD]符号在字典中下标为0）
        # 2给句子填充 or 截断，使每个句子保持固定的长度
        # 3用 “attention mask” 来显示的区分填充的 tokens 和非填充的 tokens，非填充的用1表示反之用0表示
    # 函数tokenizer.encode_plus
    # 包含以下步骤：
    # 1将句子分词为tokens。
    # 2在两端添加特殊符号[CLS]和[SEP]。
    # 3将tokens映射为下标IDs。
    # 4将列表填充或截断为固定的长度。
    # 5创建attention masks，将填充的和非填充tokens区分开来
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # 输入文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=20,  # 填充 & 截断长度
            pad_to_max_length=True,
            return_attention_mask=True,  # 返回 attn. masks.
            return_tensors='pt',  # 返回 pytorch tensors 格式的数据
        )

        # 将编码后的文本加入到列表
        input_ids.append(encoded_dict['input_ids'])

        # 将文本的 attention mask 也加入到 attention_masks 列表
        attention_masks.append(encoded_dict['attention_mask'])

    # 将列表转换为 tensor
    input_ids = torch.cat(input_ids, dim=0)#按行拼接张量
    attention_masks = torch.cat(attention_masks, dim=0)
    #labels = torch.tensor(labels)

    # 输出第 1 行文本的原始和编码后的信息
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    print('attention_mask',attention_masks)


    bert=Bert()#下载的预训练模型保存在c/user/76956/.cache文件夹中
    def test():
        a=bert(input_ids,attention_masks)
        return a
    lp = LineProfiler()
    forward=lp(test)#就是把函数装饰一下
    for i in range(100):
        forward()
    lp.print_stats(output_unit=1)#打印分
    # b=a['pooler_outp  ut']
    # b=list(b)
    # print(a)
    # print(a.shape)
    # print('the value of a:',a[1,0,:])
    # print()
    # print(a[1,0,:].shape)


