from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from .lr_scheduler import LR_Scheduler
from transformers.optimization import AdamW

def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,nesterov=True)
    elif   name=='adam':# betas用于计算梯度以及梯度平方的运行平均值的系betas1 一阶矩阵指数衰减率    在二阶beta2对于稀疏梯度（NLP方向）应设为接近1的数  如0.999
        optimizer =torch.optim.Adam(parameters,lr=lr,betas=(0.9,0.999),weight_decay=weight_decay)#,eps=1e-6   esp为了提高数值的稳定性而添加到分母的项避免分母为零#,amsgrad=True 保留历史最大的v_t，记为v_{max}，每次计算都是用最大的v_{max}，否则是用当前v_t
    elif name=='adamw':
        optimizer  =AdamW(parameters,lr=1e-5)

    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            ),
            trust_coefficient=0.001, 
            clip=False
        )
    else:
        raise NotImplementedError
    return optimizer



