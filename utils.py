from collections import Counter
from pathlib import Path
from random import random
from nltk import word_tokenize
import pandas as pd
import re
import numpy as np
import os
import json 
import torch
import os
import subprocess
import pickle
from cal_rouge import load_txt, test_rouge
from tqdm import tqdm


def gradient_noise_and_clip(parameters, device, noise_stddev=1e-3, max_clip=40.0):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    torch.nn.utils.clip_grad_norm_(parameters, max_clip)

    for p in parameters:
        noise = torch.randn(p.size()) * noise_stddev
        if device:
            noise = noise.to(device)
        p.grad.data.add_(noise)

def get_clusters(sentences_index_list, disco_labels_list):
  '''Split data into clusters based on discourse label'''
  
  disco_dict = {}
  for sentence_index, label in zip(sentences_index_list, disco_labels_list):

    if label in disco_dict:
      disco_dict[label].append(sentence_index)
    else:
      disco_dict[label] = [sentence_index]

  return disco_dict 

def make_simple_config_text(system_and_summary_paths):
    lines = []
    for system_path, summary_paths in system_and_summary_paths:
        line = "{} {}".format(system_path, " ".join(summary_paths))
        lines.append(line)
    return "\n".join(lines)


# Utility functions
def get_posweight(pos_fname, train_label_dir):

    label_dir = Path(train_label_dir)
    # print(f"label 目录{train_label_dir}")
    file_l = [path for path in label_dir.glob("*.json")]

    total_num = 0
    total_pos = 0
    for f in file_l:
        # print('f:',f)
        with f.open() as of:
            d = json.load(of)['labels']
        total_num += len(d)
        total_pos += sum(d)

    print('Compute pos weight done! There are %d sentences in total, with %d sentences as positive'%(total_num,total_pos))
    pos_weight = (total_num-total_pos)/float(total_pos)

    with open(pos_fname, 'w') as f:
        f.write(str(pos_weight))

    return torch.FloatTensor([pos_weight])

def get_all_text(train_input_dir):
    if isinstance(train_input_dir, list):
        file_l = train_input_dir
    else:
        train_input = Path(train_input_dir)
        file_l = [path for path in train_input.glob("*.json")]
    
    all_tokens = []
    for i, f in enumerate(file_l):
        if i % 20000 ==0:
            print('i:', i)

        with f.open() as of:
            d = json.load(of)
        tokens = [t for sent in d['inputs'] for t in (sent['tokens']+['<eos>'])]
        all_tokens.append(tokens)

    return all_tokens

def build_word2ind(utt_l, vocabularySize):
    print('Begin Words Counter!')
    word_counter = Counter([word for utt in utt_l for word in utt])
    print('%d words found!'%(len(word_counter)))

    vocabulary = ["<UNK>"] + [e[0] for e in word_counter.most_common(vocabularySize)]
    del word_counter

    word2index = {word:index for index,word in enumerate(vocabulary)}
    del vocabulary

    global EOS_INDEX
    EOS_INDEX = word2index['<eos>']

    return word2index

def build_volcabulary(train_input_dir,vocabularySize):
    print('Begin Words Counter!')
    # 读取文件列表
    if isinstance(train_input_dir, list):
        file_l = train_input_dir
    else:
        train_input = Path(train_input_dir)
        file_l = [path for path in train_input.glob("*.json")]
    
    word_counter = Counter()
    # 每一篇统计词频
    for f in tqdm(file_l, desc='Processing files', unit='file'):
        with f.open() as of:
            d = json.load(of)
            tokens = [t for sent in d['inputs'] for t in (sent['tokens']+['<eos>'])]
       
        word_counter.update(tokens)
    # 统计结束
    print('%d words found!'%(len(word_counter)))
    # 词频排序，生成词表
    vocabulary = ["<UNK>"] + [e[0] for e in word_counter.most_common(vocabularySize)]
    del word_counter
    # 词表
    word2index = {word:index for index,word in enumerate(vocabulary)}
    del vocabulary

    global EOS_INDEX
    EOS_INDEX = word2index['<eos>']

    return word2index
    






# Build embedding matrix by importing the pretrained glove
def getEmbeddingMatrix(dat_fname, gloveDir, word2index, embedding_dim):
    '''Refer to the official baseline model provided by SemEval.'''
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(word2index) , embedding_dim))
    for word, i in word2index.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    pickle.dump(embeddingMatrix, open(dat_fname, 'wb'))
    
    return embeddingMatrix


def get_rouge(args, hyp_pathlist, ref_pathlist):
    path_data = []
    for i in range(len(hyp_pathlist)):
        path_data.append([hyp_pathlist[i], [ref_pathlist[i]]])

    config_text = make_simple_config_text(path_data)
    config_path = './' + args.dataset + '_config'
    of = open(config_path,'w')
    of.write(config_text)
    of.close()

    summaries = []
    references = []
    for line in config_text.split("\n"):
        system_path = line.split(' ')[0]
        summary_path = line.split(' ')[1]

        summary = load_txt(system_path)
        reference = load_txt(summary_path)

        summaries.append(' '.join(summary))
        references.append(' '.join(reference))

    rouge_result = test_rouge(summaries, references, num_processes=8)

    return rouge_result
        
