import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, BertTokenizer
from configs import get_args, set_deterministic
from models import get_model
from gensim.models import KeyedVectors as Vectors
from utils import load_embedding
import random
from augmentations.text_aug import text_augmentations
from collections import OrderedDict
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
from SentEval.senteval.engine import SE


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def eval_in_train(args,embedding,padding_value,model,train_epoch,max_len,total_acc,num_nochange):




    model.eval()  # 在测试阶段禁止使用bn和dropout
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']  # , 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']#, 'STSBenchmark', 'SICKRelatedness'
        args.tasks += [ 'SST2']#'MR', 'CR', 'MPQA', 'SUBJ', 'TREC', 'MRPC'

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 2}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 4}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 2}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 10}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):#在sts里面使用用于编码数据
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        #get the embedding of sentences
        batch=torch.tensor([])
        for sentence in sentences:
            s1 = torch.tensor([])
            s2 = torch.tensor([])
            element_num = []
            sent_vector, emypt_idx = load_embedding(args.emb_format, embedding, sentence)
            # sent_vector=data_selfattention_befor_aug(sent_vector)
            # sent_vector= np.array(sent_vector, dtype=np.float32)
            if len(sent_vector) < max_len:

                sent_vector = np.row_stack((sent_vector, [padding_value] * (max_len - len(sent_vector))))
            else:

                sent_vector = sent_vector[:max_len]

            s1 = sent_vector
            s1 = torch.tensor(s1)  # sent_vector,s1

            batch = torch.cat((batch, s1.unsqueeze(0)), 0)  # .extend把任何类型的数据和之前的list连接在一起
            # mask=torch.cat((mask,element_num.unsqueeze(0)),0)

        # word2vec_averg
        # feature=torch.mean(batch,dim=1)
        # return feature.cpu()

        batch = batch.to('cuda')
        # mask=mask.to(device)#在测试中加上attention效果不好
        with torch.no_grad():
            # batch=data_selfattention(batch,batch,batch,mask)
            output = model.backbone(batch)
        return output.cpu()
    #get the results of senteval
    results = {}
    for task in args.tasks:
        se = SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)



    
    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))
        best_acc=0
        task_names_sts = []
        scores_sts = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names_sts.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores_sts.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores_sts.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores_sts.append("0.00")
        task_names_sts.append("Avg.")
        scores_sts.append("%.2f" % (sum([float(score) for score in scores_sts]) / len(scores_sts)))
        print_table(task_names_sts, scores_sts)

        #task_names_transfer = []
        #scores_transfer = []
        #for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            #task_names_transfer.append(task)
            #if task in results:
                #scores_transfer.append("%.2f" % (results[task]['acc']))
            #else:
                #scores_transfer.append("0.00")
        #task_names_transfer.append("Avg.")
        #scores_transfer.append("%.2f" % (sum([float(score) for score in scores_transfer]) / len(scores_transfer)))
        #print_table(task_names_transfer, scores_transfer)
        scores=[float(score) for score in scores_sts]
        if max(total_acc) < scores[-1]:

            print('找到最佳模型')
            num_nochange=0
            best_acc=float(scores[-1])
            model_path = os.path.join(args.output_dir, f'{args.model}-{args.data_name}-train_epoch{train_epoch+ 1}-acc{best_acc}.pth')
            torch.save({
                'state_dict': model.state_dict()
                # 'optimizer':optimizer.state_dict(), # will double the checkpoint file size
                 }, model_path)
            print(f"Model saved to {model_path}")
        else:
            num_nochange+=1
            print(f'未改进{num_nochange}best_acc{max(total_acc)}')
        
    return scores[-1],num_nochange

  



    


    














