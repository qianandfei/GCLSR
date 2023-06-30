import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models import TextCNN
from configs import get_args
from utils import data_selfattention
from torch.nn import CrossEntropyLoss



def cl_cal_loss(vec1,vec2):
    loss_for_cl=CrossEntropyLoss(ignore_index=-100)
    labels=torch.arange(0,vec1.shape[0],device='cuda')
    vec1=F.normalize(vec1, p=2, dim=1)#归一化为单位向量[bs,hiden_len]
    vec2=F.normalize(vec2, p=2, dim=1)#[bs,hiden_len]
    sims=vec1.matmul(vec2.T)*20
    loss=loss_for_cl(sims,labels)#拉近二者距离
    return loss


def cos_similarity(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=4096):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=4096, hidden_dim=1024, out_dim=4096): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimensmaking hion is 512,  a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
class attention(nn.Module):
    def __init__(self,dim_in, dim_k,dim_v):
        super().__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k)
        self.linear_k = nn.Linear(dim_in, dim_k)
        self.linear_v = nn.Linear(dim_in, dim_v)

    def forward(self,x,mask):
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        x=data_selfattention(q,k,v,mask)
        return x


class Plus_Proj_layer(nn.Module):#   继承类nn.Module
    def __init__(self, backbone):
        super().__init__()  # 调用父类的一个方法通过super（）来实现，直接用类名调用父类方法 但其父类的初始化的值无法继承
                            #构造函数__init__实例化之后相当于调用了Module（），而其中的__call__中有调用forward（）
        model_opt = TextCNN.ModelConfig()
        args=get_args()
        self.backbone = backbone #  调试后投影层有3层
        #self.att_model=nn.Linear(args.emb_dim, 512,512)
        #self.attention_data1=attention(args.emb_dim, 300, 300)
        #self.attention_data2=attention(args.emb_dim, 300, 300)
        #self.match_model = nn.Linear(1000, args.emb_dim)
        if args.backbone=='textcnn':
            self.projector = projection_MLP(model_opt.model_dim,4096)
            
        else:
            self.projector = projection_MLP(768)
        self.encoder = nn.Sequential( # f encoder
             self.backbone,
             self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2, mask):
        x1=data_selfattention(x1,x1,x1,mask)#不加attention为42.69
        x2=data_selfattention(x2,x2,x2,mask)
        z1,z2=self.encoder(x1),self.encoder(x2)
        p1,p2=self.predictor(z1),self.predictor(z2)
        return p1,z2,p2,z1






if __name__ == "__main__":
    model = Plus_Proj_layer()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)#创建像x1的大小的张量

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(cos_similarity(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(cos_similarity(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












