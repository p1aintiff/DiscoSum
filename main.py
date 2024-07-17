from torch import nn
from collections import Counter
from nltk import word_tokenize
from torch.autograd import Variable
import pandas as pd
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.utils import shuffle
import json 
import argparse
from data import *
from utils import *
from run import *
from models import *
from timeit import default_timer as timer

from torch.utils import tensorboard



# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cell", default='gru', help="Choose one from gru, lstm")
parser.add_argument("--embedding_dim", type=int, default=300, help="Set the dimension of word_embedding")
parser.add_argument("--hidden_dim", type=int, default=300, help="Set the dimension of hidden state")
parser.add_argument("--discourse_dim", type=int, default=32, help="Set the dimension of hidden state")
parser.add_argument("--section_dim", type=int, default=32, help="Set the dimension of hidden state")
parser.add_argument("--content_size", type=int, default=7, help="Set the dimension of hidden state")
parser.add_argument("--mlp_size", type=int, default=100, help="Set the dimension of the integrated mlp layer")
parser.add_argument("--num_epoch", type=int, default=50, help="Set the number of epoch")
parser.add_argument("--batchsize", type=int, default=64, help="Set the size of batch")
parser.add_argument("--gloveDir", type=str, default='./', help="Directory storing glove embedding")
parser.add_argument("--refpath", type=str, default='human-abstracts', help="Directory storing human abstracts")
parser.add_argument("--vocab_size", type=int, default=50000, help="vocabulary size")
parser.add_argument("--device", type=int, default=3, help="device used to compute")
parser.add_argument("--seed", type=int, default=1, help="Set the seed of pytorch, so that you can regenerate the result.")
parser.add_argument("--length_limit", type=int, default=200, help="length limit of extractive summarization")

parser.add_argument("--dataset", type=str, default='arxiv', help=['arxiv','pubmed'])
parser.add_argument("--train_input", type=str, default=None, help="The filepath of input of training data")
parser.add_argument("--train_label", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--train_abstract_discourse", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--train_content_discourse", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--val_input", type=str, default=None, help="The filepath of input of validation data")
parser.add_argument("--val_label", type=str, default=None, help="The filepath of label of validation data")
parser.add_argument("--val_abstract_discourse", type=str, default = None, help="The filepath of label of validation data")
parser.add_argument("--val_content_discourse", type=str, default=None, help="The filepath of label of training data")
parser.add_argument("--model", type=str, default='ext_summ', help=['ext_summ', 'sent_discourse_summ', 'sent_sect_discourse_summ', 'sent_label_discourse_sum'])
parser.add_argument("--runtime", type=str, default=0, help="Index of this model")
parser.add_argument("--mode", type=str, default='test', help="test or validate")
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="teacher_forcing_ratio")
parser.add_argument('--train_from', default='', type=str)
parser.add_argument('--sample_size', default=0, type=int)

args = parser.parse_args()
print(args)


# Set the global variables
HIDDEN_DIM = args.hidden_dim
NUM_EPOCH = args.num_epoch
BATCH = args.batchsize
NUM_RUN = args.runtime
EMBEDDING_DIM = args.embedding_dim
DISCOURSE_DIM = args.discourse_dim
SECTION_DIM = args.section_dim
MLP_SIZE = args.mlp_size
CELL_TYPE = args.cell
LENGTH_LIMIT = args.length_limit
VOCABULARY_SIZE = args.vocab_size
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio

LEARNING_RATE = 1e-4
TEACHER_FORCING = False
model_name = args.model

# if seed is given, set the seed for pytorch on both cpu and gpu
if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# reference path and the temorary path to store the generated summaries of validation set
ref_path = './datasets/' + args.dataset + '/' + args.refpath
hyp_path = './eval_hyp/%s-%s-%s/'%(args.dataset, args.model, args.runtime)
if not os.path.exists(hyp_path):
    os.makedirs(hyp_path)

# set the directory to store models, make new if not exists
MODEL_DIR = './pretrained_models/' + args.model + '-' + args.dataset + '-' + NUM_RUN
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# set the device the model running on
device = torch.device("cuda:%d"%(args.device))
torch.cuda.set_device(args.device)


# set the training and validation directories
train_input_dir = './datasets/' + args.dataset + '/' + args.train_input
train_label_dir = './datasets/' + args.dataset + '/' + args.train_label
train_abstract_discourse_dir = './datasets/' + args.dataset + '/' + args.train_abstract_discourse
train_content_discourse_dir = './datasets/' + args.dataset + '/' + args.train_content_discourse

val_input_dir = './datasets/' + args.dataset + '/' + args.val_input
val_label_dir = './datasets/' + args.dataset + '/' + args.val_label
val_abstract_discourse_dir = './datasets/' + args.dataset + '/' + args.val_abstract_discourse
val_content_discourse_dir = './datasets/' + args.dataset + '/' + args.val_content_discourse

# build the vocabulary dictionary
if 'vocabulary_%s.json'%(args.dataset) in [path.name for path in Path('./').glob('*.json')]:
    with open('vocabulary_%s.json'%(args.dataset),'r') as f:
        w2v = json.load(f)
    print('Vocabulary exists, Load vocabulary from vocabulary_%s.json'%(args.dataset))
else: 
    print('Begin Build vocabulary from vocabulary_%s.json'%(args.dataset))
    # all_tokens = get_all_text(train_input_dir)
    # w2v = build_word2ind(all_tokens, VOCABULARY_SIZE)
    w2v = build_volcabulary(train_input_dir, VOCABULARY_SIZE)
    with open('vocabulary_%s.json'%(args.dataset),'w') as f:
        json.dump(w2v,f)
    print('Load vocabulary from vocabulary_%s.json'%(args.dataset))
sys.stdout.flush()

#------set weight manually
pos_weight = torch.tensor(1.0).to(device)
print('pos_weight:',pos_weight)

# build embedding matrix
dat_fname = args.dataset + '_embeddingMatrix.dat'
if os.path.exists(dat_fname):
    print('Embedding_matrix exist:', dat_fname)
    embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    print('embedding_matrix_size loaded:', len(embedding_matrix))
else:
    print('loading embedding_matrix...')
    embedding_matrix = getEmbeddingMatrix(dat_fname, args.gloveDir, w2v, EMBEDDING_DIM)


# set the dataset and dataloader for both training and validation set.
train_dataset = SummarizationDataset(w2v, embedding_matrix, EMBEDDING_DIM, train_input_dir, train_abstract_discourse_dir, train_content_discourse_dir, target_dir=train_label_dir,subset_size=args.sample_size)
train_dataloader = SummarizationDataLoader(train_dataset, args.content_size, batch_size=BATCH)

# check_loader(train_dataloader)


val_dataset = SummarizationDataset(w2v, embedding_matrix, EMBEDDING_DIM, val_input_dir, val_abstract_discourse_dir, val_content_discourse_dir, target_dir=val_label_dir, reference_dir=ref_path)
val_dataloader = SummarizationDataLoader(val_dataset, args.content_size, batch_size=BATCH)

del embedding_matrix,w2v,train_dataset,val_dataset

# Initialize the model
if args.model == 'ext_summ':
    model = Ext_summ(EMBEDDING_DIM, HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
elif args.model == 'multi_sent_discourse_summ':
    model = Multi_Sent_Discourse_summ(EMBEDDING_DIM, HIDDEN_DIM, DISCOURSE_DIM, SECTION_DIM, MLP_SIZE, TEACHER_FORCING_RATIO, cell_type=CELL_TYPE)
elif args.model == 'ext_emb_summ':
    model = Ext_Emb_summ(EMBEDDING_DIM, HIDDEN_DIM, DISCOURSE_DIM, SECTION_DIM, MLP_SIZE, cell_type=CELL_TYPE)

if torch.cuda.is_available():
    model = model.to(device)

sys.stdout.flush()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
best_r = 0
best_ce = 1000
train_loss=[]
val_loss = []
print('Start Training!')

time_start = timer()
time_epoch_end_old = time_start

if args.train_from != '':
    print("train from : {}".format(args.train_from))
    checkpoint = torch.load(args.train_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

writer = tensorboard.SummaryWriter(log_dir="/root/tf-logs",comment=str(args.model),filename_suffix="num_epoch"+str(args.num_epoch))
for epoch in range(NUM_EPOCH):

    l = train_seq2seq(args, train_dataloader, model, optimizer, pos_weight, device, model_name, TEACHER_FORCING,epoch_idx=epoch, writer=writer)
    train_loss.append(l)
    print('Epoch %d finished, the avg loss: %f'%(epoch,l))

    r, l = eval_seq2seq(args, val_dataloader, val_content_discourse_dir, model, hyp_path, LENGTH_LIMIT, pos_weight, device, model_name)
    val_loss.append(l)
    print('Epoch %d finished '%(epoch))
    print('Validation loss: %f'%(l))
    print('Rouge-ave f1: %f'%(r))

    if r > best_r:
        PATH = MODEL_DIR+'/' + args.model + '_' + str(epoch) + '_best_r'
        best_r = r
        torch.save(model.state_dict(), PATH)
        print('Epoch %d, saved as best model - highest r2%f.'%(epoch, best_r))

        checkpoint = {'epoch': epoch, 'optim': optimizer.state_dict(), 'model': model.state_dict()}
        torch.save(checkpoint, MODEL_DIR + '/' + model_name+'-Best-ROUGE-'+str(epoch))
        print('Epoch %d, saved as best checkpoint - highest r2%f.'%(epoch, best_r))

    if l <= best_ce:
        best_ce = l
        print('Epoch %d, lowest ce!'%(epoch))

    time_epoch_end_new = timer()
    print ('Seconds to execute to whole epoch: ' + str(time_epoch_end_new - time_epoch_end_old))
    time_epoch_end_old = time_epoch_end_new
    sys.stdout.flush()

print('Seconds to execute to whole training procedure: ' + str(time_epoch_end_old - time_start))
