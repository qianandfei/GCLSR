import argparse
import os
import numpy as np
import torch
import random



def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn

    else:
        print("Non-deterministic")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")

    parser.add_argument('--debug', action='store_true')
    # training specific args
    parser.add_argument('--data_name', type=str, default='wiki', help='choose from random, stl10, mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_path', type=str, default='./selfdatasets/')#针对于所使用的文件  不是此文件所对应的相对路径
    parser.add_argument('--output_dir', type=str, default='./output_data/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')



    parser.add_argument('--emb_file', type=str, default='./selfdatasets/GoogleNews-vectors-negative300.bin')#GoogleNews-vectors-negative300.bin--glove.840B.300d.txt
    parser.add_argument('--emb_format', type=str, default='word2vec')
    parser.add_argument('--min_word_freq', type=int, default=1)#建立词表最小词频
    parser.add_argument('--max_len', type=int, default=64)#padding后句子统一长度64
    parser.add_argument('--do_train', type=bool, default=True)  # padding后句子统一长度
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--hide_progress', action='store_true')

    # model related params

    parser.add_argument('--model', type=str, default='simsiam')
    parser.add_argument('--backbone', type=str, default='textcnn')
    parser.add_argument('--num_epochs', type=int, default=20, help='This will affect learning rate decay')
    parser.add_argument('--logistic_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--proj_layers', type=int, default=None, help="number of projector layers. In cifar experiment, this is set to 2")
    # optimization params
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--warmup_lr', type=float, default=0, help='Initial war mup learning rate')
    parser.add_argument('--base_lr', type=float, default=0.03)#0.04
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    args = parser.parse_args()
    
    if args.debug:
        args.batch_size = 2 
        args.stop_at_epoch = 2
        args.num_epochs = 3 # train only one epoch
        args.num_workers = 0
    assert not None in [args.output_dir, args.data_path]
    os.makedirs(args.output_dir, exist_ok=True)
    # assert args.stop_at_epoch <= args.num_epochs
    return args
