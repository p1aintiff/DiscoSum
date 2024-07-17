from torch import nn
from collections import Counter
from random import random
from nltk import word_tokenize
from torch.autograd import Variable
import pandas as pd
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.utils import shuffle
import json 
import random
import argparse
from data import *
from utils import *
from run import *
from models import *
import sys


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cell", default='gru', help="Choose one from gru, lstm")
parser.add_argument("--embedding_dim", type=int, default=300, help="Set the dimension of word_embedding")
parser.add_argument("--hidden_dim", type=int, default=300, help="Set the dimension of hidden state")
parser.add_argument("--discourse_dim", type=int, default=32, help="Set the dimension of hidden state")
parser.add_argument("--section_dim", type=int, default=32, help="Set the dimension of hidden state")
parser.add_argument("--mlp_size", type=int, default=100, help="Set the dimension of the integrated mlp layer")
parser.add_argument("--batchsize", type=int, default=128, help="Set the size of batch")
parser.add_argument("--length_limit", type = int, default=290, help="length limit of extractive summarization")
parser.add_argument("--mode", type=str, default='test', help="test or validate")
parser.add_argument("--content_size", type=int, default=7, help="Set the dimension of hidden state")
parser.add_argument("--section_size", type=int, default=15, help="Set the dimension of hidden state")
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="teacher_forcing_ratio")

parser.add_argument("--dataset", type=str, default='arxiv', help=['arxiv','pubmed'])
parser.add_argument("--test_input", type=str, default=None, help="The filepath of input of validation data")
parser.add_argument("--test_label", type=str, default=None, help="The filepath of label of validation data")
parser.add_argument("--test_file_list", type=str, default=None, help="The file storing the ids of a subset of test data")
parser.add_argument("--test_abstract_discourse", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--test_section_label", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--test_content_discourse", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--epoch", type=int, default=50, help="test or validate")

parser.add_argument("--model", type = str, default='ext_summ', help=['ext_summ', 'Sent_Discourse_summ', 'sent_sec_content_summ'])
parser.add_argument("--runtime", type=str, default=0, help="Index of this model")
parser.add_argument("--gloveDir", type=str, default='./', help="Directory storing glove embedding")
parser.add_argument("--refpath", type=str, default='human-abstracts', help="Directory storing human abstracts")
parser.add_argument("--model_path", type=str, default='pretrained_models/', help="The path of model")
parser.add_argument("--device", type=int, default=1, help="device used to compute")
parser.add_argument("--result_file_name", type=str, default=None, help="The file storing all the rouge results of test data")
parser.add_argument("--subset_size", type=int, default=0, help="subset of test data")
args = parser.parse_args()
print(args)


# Set the global variables
HIDDEN_DIM = args.hidden_dim
BATCH = args.batchsize
NUM_RUN = args.runtime
EMBEDDING_DIM = args.embedding_dim
DISCOURSE_DIM = args.discourse_dim
SECTION_DIM = args.section_dim
MLP_SIZE = args.mlp_size
CELL_TYPE = args.cell
LENGTH_LIMIT = args.length_limit
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio

LEARNING_RATE = 1e-4
USE_SECTION_INFO = False
# todo 获取目录下
# import os
# import re

# # 假设你的目录路径存储在变量dir_path中
# dir_path = '/path/to/your/model/directory'  # 请替换为实际的目录路径

# # 使用os.listdir列出目录中的所有文件和文件夹
# files_and_dirs = os.listdir(dir_path)

# # 初始化最大epoch变量
# max_epoch = 0

# # 遍历文件和文件夹
# for item in files_and_dirs:
#     # 使用正则表达式匹配文件名中的epoch数字
#     match = re.search(r'_(\d+)_best_r', item)
#     if match:
#         # 从文件名中提取epoch数字
#         current_epoch = int(match.group(1))
#         # 更新最大epoch
#         if current_epoch > max_epoch:
#             max_epoch = current_epoch

# print(f"最大的epoch是: {max_epoch}")

# # 使用找到的最大epoch更新MODEL_PATH
# MODEL_PATH = os.path.join(dir_path, args.model + '-' + args.dataset + '-' + NUM_RUN + '/' + args.model + '_' + str(max_epoch) + '_best_r')
# print(f"更新后的MODEL_PATH是: {MODEL_PATH}")

MODEL_PATH = args.model_path + args.model + '-' + args.dataset + '-' + NUM_RUN + '/' + args.model + '_' + str(args.epoch) + '_best_r'
SAVE_RESULT_NAME = args.result_file_name

model_name = args.model

# Set the refpath (human-abstraction) and hyp-path(to store the generated summary)
ref_path = './datasets/' + args.dataset + '/' + args.refpath
hyp_path = './test_hyp/%s-%s-%s/'%(args.dataset, args.model, args.runtime)
if not os.path.exists(hyp_path):
    os.makedirs(hyp_path)

device = torch.device("cuda:%d"%(args.device))
torch.cuda.set_device(args.device)


# If the test is on a subset of the whole dataset
test_input_dir =  './datasets/' + args.dataset + '/' + args.test_input
test_label_dir =  './datasets/' + args.dataset + '/' + args.test_label
test_abstract_discourse_dir = './datasets/' + args.dataset + '/' + args.test_abstract_discourse
test_content_discourse_dir = './datasets/' + args.dataset + '/' + args.test_content_discourse

# build the vocabulary dictionary
if 'vocabulary_%s.json'%(args.dataset) in [path.name for path in Path('./').glob('*.json')]:
    with open('vocabulary_%s.json'%(args.dataset),'r') as f:
        w2v = json.load(f)
    print('Load vocabulary from vocabulary_%s.json'%(args.dataset))
else: 
    print('Begin Build vocabulary from vocabulary_%s.json'%(args.dataset))
    all_tokens = get_all_text(args.train_input_dir)
    w2v = build_word2ind(all_tokens, args.VOCABULARY_SIZE)
    with open('vocabulary_%s.json'%(args.dataset),'w') as f:
        json.dump(w2v,f)
    print('Load vocabulary from vocabulary_%s.json'%(args.dataset))
sys.stdout.flush()

# Get the postive weight to compute the weighted loss
print('Calculate get_posweight...')
pos_fname = args.dataset + '_pos_weight.txt'
if os.path.exists(pos_fname):
    with open(pos_fname, 'r') as f:
        for line in f:
            pos_weight = float(line.strip())

    if torch.cuda.is_available():
        pos_weight = torch.FloatTensor([pos_weight]).to(device)
        print('pos_weight:',pos_weight)
else:
    # pos_weight = get_posweight(pos_fname, args.test_label)
    pos_weight = get_posweight(pos_fname, test_label_dir)
    if torch.cuda.is_available():
        pos_weight = pos_weight.to(device)
    print('pos_weight:',pos_weight)

# build embedding matrix
dat_fname = args.dataset + '_embeddingMatrix.dat'
if os.path.exists(dat_fname):
    print('loading embedding_matrix:', dat_fname)
    embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    print('embedding_matrix_size loaded:', len(embedding_matrix))
else:
    print('loading word vectors...')
    embedding_matrix = getEmbeddingMatrix(dat_fname, args.gloveDir, w2v, EMBEDDING_DIM)

# Build the dataset
test_dataset = SummarizationDataset(w2v, embedding_matrix, EMBEDDING_DIM, test_input_dir, test_abstract_discourse_dir, test_content_discourse_dir, target_dir=test_label_dir, reference_dir=ref_path,subset_size=args.subset_size)
test_dataloader = SummarizationDataLoader(test_dataset, args.content_size, batch_size=BATCH, shuffle=False)


print('Start loading model.')
# Initialize the model
if args.model == 'ext_summ':
    model = Ext_summ(EMBEDDING_DIM, HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
elif args.model == 'multi_sent_discourse_summ':
    model = Multi_Sent_Discourse_summ(EMBEDDING_DIM, HIDDEN_DIM, DISCOURSE_DIM, SECTION_DIM, MLP_SIZE, TEACHER_FORCING_RATIO, cell_type=CELL_TYPE)
elif args.model == 'ext_emb_summ':
    model = Ext_Emb_summ(EMBEDDING_DIM, HIDDEN_DIM, DISCOURSE_DIM, SECTION_DIM, MLP_SIZE, cell_type=CELL_TYPE)
    
# Load the pre-trained model
# 手动输入
# model_self_path = "./pretrained_models/multi_sent_discourse_summ-pubmed-both-8-64-32-32-1.5-1/multi_sent_discourse_summ_28_best_r"
# model.load_state_dict(torch.load(model_self_path))
model.load_state_dict(torch.load(MODEL_PATH))

# Move to GPU
if torch.cuda.is_available():
    model = model.to(device)

# We also want the Rouge-L score
lcs = True
model.eval()

print('Start evaluating.')
r2, l = eval_seq2seq(args, test_dataloader, test_content_discourse_dir, model, hyp_path, LENGTH_LIMIT, pos_weight, device, model_name)
print('test loss: %f'%(l))
sys.stdout.flush()
