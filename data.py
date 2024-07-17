from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import json
import utils
import torch
from models import *
import collections
import numpy as np 
import random


discourse_label_list = ['PAD_label', 'BAC_label', 'OBJ_label', 'APP_label', 'OUT_label', 'OTH_label']
discourse_mappings = {discourse_label : i for i, discourse_label in enumerate(discourse_label_list, 0)}
# {'PAD_label': 0, 'BAC_label': 1, 'OBJ_label': 2, 'APP_label': 3, 'OUT_label': 4, 'OTH_label': 5}
# {0:'PAD_label', 1:'BAC_label', 2:'OBJ_label', 3:'APP_label', 4:'OUT_label', 5:'OTH_label'}

section_label_list = ['pad', 'introduction', 'conclusion', 'result', 'discussion', 'model', 'method', 'background', 'other']
section_mappings = {section_label : i for i, section_label in enumerate(section_label_list, 0)}
# {'pad': 0, 'introduction': 1, 'conclusion': 2, 'result': 3, 'discussion': 4, 'model': 5, 'method': 6, 'background': 7, 'other': 8}


def section2index(section_names):
    section_names_label = []

    for name in section_names:
        if 'intro' in name:
            name_label = 'introduction'
        elif 'conclu' in name or 'summary' in name:
            name_label = 'conclusion'
        elif 'result' in name or 'analys' in name or 'ablat' in name:
            name_label = 'result'
        elif 'discus' in name:
            name_label = 'discussion'
        elif 'architec' in name or 'system' in name or 'model' in name:
            name_label = 'model'
        elif 'method' in name or 'implement' in name:
            name_label = 'method'
        elif 'background' in name or 'motivation' in name:
            name_label = 'background'
        else:
            name_label = 'other'

        section_names_label.append(section_mappings[name_label])

    return section_names_label

def get_lens(batch):
    return [len(item) for item in batch]

def padding_a_discourse(discourse, pad_token, max_len):
    '''
    pad input to max length
    '''
    
    lens = get_lens(discourse)
    max_len = min(max(lens), max_len)
    # max_len = max(lens)

    discourse_padded = [] 
    for i, l in enumerate(lens):
        if l > max_len:
            l = max_len
        discourse_padded.append(discourse[i][:l] + [pad_token] * (max_len - l))
        
    return discourse_padded

class SummarizationDataset(Dataset):
    def __init__(self, word2index, embedding_matrix, embedding_size, input_dir, abstract_discourse_dir, content_discourse_dir, target_dir=None, reference_dir=None,subset_size=0):
        self._w2i = word2index

        inputs_dir = Path(input_dir)
        self._inputs = [path for path in inputs_dir.glob("*.json")]
        if subset_size > 0:
            self._inputs = random.sample(self._inputs, subset_size)
        print('inputs的数量',len(self._inputs))
        self._inputs.sort()
        self._abstract_discourse_dir = None
        self._target_dir = None
        self._reference_dir = None
        self.embedding_matrix = embedding_matrix

        if abstract_discourse_dir:
            self._abstract_discourse_dir = Path(abstract_discourse_dir)

        if content_discourse_dir:
            self._content_discourse_dir = Path(content_discourse_dir)

        if target_dir:
            self._target_dir = Path(target_dir)

        if reference_dir:
            self._reference_dir = reference_dir

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):

        p = self._inputs[idx]
        out = {}
        try:
            with p.open() as of: 
                data = json.load(of)
        except:
            print('error:',idx)
            print('文件路径',self._inputs[idx])
            return self.__getitem__(idx-1)

        out['id'] = data['id']
        out['filename'] = p
        # print(data['id'])
        
        # Document_l is a list of list of word indexes, each sublist is a sentence, and each sentence is end with a <eos>
        document_l = []
        for i in data['inputs']:
            sent_l = []
            for w in i['tokens']:
                sent_l.append(self._w2i.get(w, 0))  # ["<UNK>"]=0

            sent_embed = torch.FloatTensor(self.embedding_matrix[sent_l,:])  # [number_of_words, word_dim]
            document_l.append(sent_embed)
        # print('document_l:',len(document_l))

        # Truncate by number of sentences
        # out['document'] = document_l[0:2000]  # !!!!!!!!!!!!!!!!!!  arxiv:3000, pubmed:2500; 
        # out['num_sentences'] = len(out['document'])

        # default length
        out['document'] = document_l 
        out['num_sentences'] = len(out['document'])

        # section_names labels       
        out['section_lengths'] = data['section_lengths']
        section_names_list = section2index(data['section_names'])

        section_lengths = data['section_lengths']
        section_names_labels = []
        for label, section_length in zip(section_names_list, section_lengths):
            labels = [label] * section_length
            section_names_labels.extend(labels)
        out['sentence_names'] = section_names_labels[0:out['num_sentences']]

        # abstract-discourse 
        abstract_discourse_file = self._abstract_discourse_dir / "{}.json".format(out["id"])
        with abstract_discourse_file.open() as of:
            abstract_discourse_data = json.load(of)
        # print('abstract_discourse_file:',abstract_discourse_file)
        # print('abstract_discourse_data:',abstract_discourse_data)
        out['abstract_discourse'] = [discourse_mappings[discourse] if discourse != None else 0 for discourse in abstract_discourse_data]

        # content-discourse 
        content_discourse_file = self._content_discourse_dir / "{}.json".format(out["id"])
        with content_discourse_file.open() as of:
            content_discourse_data = json.load(of)
        # print('content_discourse_file:',content_discourse_file)
        # print('content_discourse_data:',content_discourse_data)
        out['content_discourse'] = [discourse_mappings[discourse] if discourse != None else 0 for discourse in content_discourse_data][0:out['num_sentences']] # !!!!!!!!!!!!!!!!!!

        # If targets are given, then read the targets
        out['labels'] = None
        # print(self._target_dir,end='')
        if self._target_dir:
            target_file = self._target_dir / "{}.json".format(out["id"])
            if target_file.exists():
                try:
                    with target_file.open() as of:
                        label_data = json.load(of)
                    out['labels'] = label_data['labels']

                    if label_data['labels'] != []:
                        out['labels'] = label_data['labels']
                    else:
                        out['labels'] = [0]*out['num_sentences']
                except:
                    out['labels'] = [0]*out['num_sentences']
        else:
            print('no label')
        out['labels'] = out['labels'][0:out['num_sentences']] # !!!!!!!!!!!!!!!!!!

        # If the reference is given, load the reference
        out['reference'] = None
        if self._reference_dir:
            ref_file = self._reference_dir +"{}.txt".format(out["id"])
            out['reference'] = ref_file

        assert len(out['labels']) == len(out['document'])
        # print('labels:',len(out['labels']))
        # print('document:',len(out['document']))
        # print('num_sentences:',out['num_sentences'])
        # print('id:',out['id'])

        return out

class SummarizationDataLoader(DataLoader):
    def __init__(self, dataset, content_size, batch_size=1, shuffle=True):
        super(SummarizationDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.avgsent_batch)

        self.content_size = content_size

    # def avgsent_batch(self, batch):
    #     batch.sort(key=lambda x: x["num_sentences"], reverse=True)

    #     out = {}
    #     out['id'] = []
    #     out['refs'] = []
    #     out['filenames'] = []
    #     out['discourses'] = []

    #     doc_batch = []
    #     labels_batch = []
    #     names_batch = []
    #     doc_lengths = []
    #     abstract_discourse_batch = []
    #     content_discourse_batch = []
    #     for d in batch:
    #         out['id'].append(d['id'])
    #         abstract_discourse_batch.append(d['abstract_discourse'])

    #         doc_l = torch.FloatTensor(d['num_sentences'], d['document'][0].size()[1])  
    #         for i in range(len(d['document'])): 
    #             doc_l[i,:] = torch.mean(d['document'][i], 0)   # sent_emb; d['document'][i] = [number_of_words, word_dim]

    #         doc_batch.append(doc_l)
    #         labels_batch.append(torch.FloatTensor(d['labels']).unsqueeze(1))
    #         names_batch.append(torch.LongTensor(d['sentence_names']))
    #         content_discourse_batch.append(torch.LongTensor(d['content_discourse']))
    #         doc_lengths.append(d['num_sentences'])

    #         out['filenames'].append(d['filename'])
    #         if d['reference'] != None:
    #             out['refs'].append(d['reference'])

    #     padded_labels_batch = pad_sequence(labels_batch, padding_value=-1)
    #     padded_names_batch = pad_sequence(names_batch, padding_value=0)
    #     padded_content_discourse_batch = pad_sequence(content_discourse_batch, padding_value=0)
    #     padded_doc_batch = pad_sequence(doc_batch, padding_value=-1)
    #     # packed_padded_doc_batch = pack_padded_sequence(padded_doc_batch, doc_lengths)

    #     # out['document'] = packed_padded_doc_batch
    #     out['document'] = padded_doc_batch
    #     out['labels'] = padded_labels_batch
    #     out['names'] = padded_names_batch
    #     out['input_length'] = torch.LongTensor(doc_lengths)
    #     out['doc_lengths'] = doc_lengths
    #     out['abstract_discourses'] = torch.LongTensor(padding_a_discourse(abstract_discourse_batch, pad_token=0, max_len=self.content_size))
    #     out['content_discourses'] = padded_content_discourse_batch



    #     # 检查是否有nan
    #     has_nan=torch.isnan(out['document'])
    #     nam_num = torch.sum(has_nan).item()
    #     if  nam_num> 0:
    #         print('inputs:',nam_num)
    #         print('inputs:',has_nan)
    #         # 找到包含NaN值的位置
    #         nan_indices = torch.nonzero(has_nan).squeeze()

    #         # 输出结果
    #         print("Indices of NaN values:", nan_indices)
    #         input('nan in inputs')
    #     return out
        
        
    def avgsent_batch(self, batch):
        batch.sort(key=lambda x: x["num_sentences"], reverse=True)

        out = {}
        out['id'] = []  # 添加一个字段用于存储文档的ID
        out['refs'] = []
        out['filenames'] = []
        out['discourses'] = []

        doc_batch = []
        labels_batch = []
        names_batch = []
        doc_lengths = []
        abstract_discourse_batch = []
        content_discourse_batch = []
        for d in batch:
            out['id'].append(d['id'])  # 将每个文档的ID添加到列表中
            abstract_discourse_batch.append(d['abstract_discourse'])

            doc_l = torch.FloatTensor(d['num_sentences'], d['document'][0].size()[1])  
            for i in range(len(d['document'])): 
                doc_l[i,:] = torch.mean(d['document'][i], 0)   # sent_emb; d['document'][i] = [number_of_words, word_dim]

            doc_batch.append(doc_l)
            labels_batch.append(torch.FloatTensor(d['labels']).unsqueeze(1))
            names_batch.append(torch.LongTensor(d['sentence_names']))
            content_discourse_batch.append(torch.LongTensor(d['content_discourse']))
            doc_lengths.append(d['num_sentences'])

            out['filenames'].append(d['filename'])
            if d['reference'] != None:
                out['refs'].append(d['reference'])

        padded_labels_batch = pad_sequence(labels_batch, padding_value=-1)
        padded_names_batch = pad_sequence(names_batch, padding_value=0)
        padded_content_discourse_batch = pad_sequence(content_discourse_batch, padding_value=0)
        padded_doc_batch = pad_sequence(doc_batch, padding_value=-1)

        out['document'] = padded_doc_batch
        out['labels'] = padded_labels_batch
        out['names'] = padded_names_batch
        out['input_length'] = torch.LongTensor(doc_lengths)
        out['doc_lengths'] = doc_lengths
        out['abstract_discourses'] = torch.LongTensor(padding_a_discourse(abstract_discourse_batch, pad_token=0, max_len=self.content_size))
        out['content_discourses'] = padded_content_discourse_batch

        # 检查是否有nan
        has_nan=torch.isnan(out['document'])
        nam_num = torch.sum(has_nan).item()
        if  nam_num > 0:
            print('inputs:', nam_num)
            print('inputs:', has_nan)
            # 找到包含NaN值的位置
            nan_indices = torch.nonzero(has_nan).squeeze()

           # 输出结果以及对应的ID
            nan_indices_list = nan_indices.tolist()  # 将张量转换为 Python 列表
            print("Indices of NaN values:", nan_indices_list)
            # print("IDs of documents with NaN values:", [out['id'][i] for i in nan_indices_list])
            input('nan in document')

            # 使用 pickle 将字典序列化为字节流
            with open("out.pkl", "wb") as f:
                pickle.dump(out, f)
            
            with open("batch.pkl", "wb") as f:
                pickle.dump(batch, f)
            
            with open("nan_indices_list.pkl", "wb") as f:
                pickle.dump(nan_indices_list, f)


        return out
