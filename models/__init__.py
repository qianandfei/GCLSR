from .plus_proj_layer import Plus_Proj_layer

from torchvision.models import resnet50, resnet18
import torch
from .backbones import *

def get_backbone(backbone):
    if backbone=='textcnn':
        backbone=textcnn()
    if backbone=='bert':
        backbone=Bert()
    else:
        NotImplementedError
    # if castrate:
    #     backbone.output_dim = backbone.fc.in_features
    #     backbone.fc = torch.nn.Identity()#nn.Identity()为站位符 什么都不做 一般用于输入
    return backbone


def get_model(name, backbone):
    if name == 'bert_cnn':
        model = Plus_Proj_layer(get_backbone(backbone))
    elif name == 'bert_lstm':
        model = Bert_Lstm(get_backbone(backbone))
    elif name == 'simsiam':
        model = Plus_Proj_layer(get_backbone(backbone))
    else:
        raise NotImplementedError
    return model






