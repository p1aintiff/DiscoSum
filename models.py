from collections import Counter
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from random import random

from utils import *

discourse_mappings = {'PAD_label': 0, 'BAC_label': 1, 'OBJ_label': 2, 'APP_label': 3, 'OUT_label': 4, 'OTH_label': 5}
reverse_discourse_mappings = {0:'PAD_label', 1:'BAC_label', 2:'OBJ_label', 3:'APP_label', 4:'OUT_label', 5:'OTH_label'}

section_mappings = {'pad': 0, 'introduction': 1, 'conclusion': 2, 'result': 3, 'discussion': 4, 'model': 5, 'method': 6, 'background': 7, 'other': 8}


# Only use sentence representation
class Ext_summ(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_size, cell_type='gru'):
        super(Ext_summ, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type

        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True)

        self.hidden2out = self.build_mlp(hidden_size*2, mlp_size, 0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.dropout_layer = nn.Dropout(p=0.3)

    def build_mlp(self, input, output, dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU(inplace=False))
        mlp.append(nn.Dropout(p=dropout, inplace=False))
        return nn.Sequential(*mlp)

    def forward(self, inputs, doc_lengths, device):

        inputs = pack_padded_sequence(inputs, doc_lengths)
        output, hidden = self.rnn(inputs)        
        # print('output:',output.size())
        # print('hidden:',hidden.size())

        output, _ = pad_packed_sequence(output)
        output = self.dropout_layer(output)  # output = [seq_len, batch, hidden_size*2]
        # print('inputs:',inputs.size())

        mlp_out = self.hidden2out(output)  # mlp_out = [seq_len, batch, mlp_size]
        out = self.final_layer(mlp_out)    # out = [seq_len, batch, 1]

        return out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames, hyp_path):
        #score_batch = [batch, seq_len]
        summaryfile_batch = []
        all_ids = []

        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _, x in sorted(zip(scores, list(range(input_lengths[i]))), reverse=True)]
            fn = filenames[i]

            with fn.open() as of:
                inputs = json.load(of)['inputs']

            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc += inputs[j]['word_count']
                if wc >= length_limit:
                    break
            summary = '\n'.join(summary)

            fname = hyp_path + ids[i] + '.txt'
            of = open(fname,'w')
            of.write(summary)

            all_ids.append(selected_ids)
            summaryfile_batch.append(fname)

        return summaryfile_batch, all_ids


# Multi_Sent_Discourse_summ
class Multi_Sent_Discourse_summ(nn.Module):
    def __init__(self, input_size, hidden_size, discourse_dim, section_dim, mlp_size, teacher_forcing_ratio, cell_type='gru'):
        super(Multi_Sent_Discourse_summ, self).__init__()
        self.hidden_size = hidden_size
        self.discourse_dim = discourse_dim
        self.section_dim = section_dim
        self.cell = cell_type
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if self.cell == 'gru':
            # self.encoder_rnn_sent = nn.GRU(input_size+self.section_dim, hidden_size, num_layers=1, bidirectional=True)
            # self.encoder_rnn_sent = nn.GRU(input_size+self.discourse_dim, hidden_size, num_layers=1, bidirectional=True)
            self.encoder_rnn_sent = nn.GRU(input_size+self.discourse_dim+self.section_dim, hidden_size, num_layers=1, bidirectional=True)
            self.decoder_rnn = nn.GRU(hidden_size*2+self.discourse_dim, hidden_size*2, num_layers=1, bidirectional=False)
        else:
            self.encoder_rnn = nn.LSTM(input_size+self.discourse_dim, hidden_size, num_layers=1, bidirectional=True)
            self.decoder_rnn = nn.LSTM(hidden_size*2+self.discourse_dim, hidden_size*2, num_layers=1, bidirectional=False)

        # 摘要结构词嵌入模型初始化                                    词数量             词嵌入的维度
        self.discourse_embed = torch.nn.Embedding(len(discourse_mappings), self.discourse_dim)
        # 这个嵌入是自己定义的，所以需要训练
        self.discourse_embed.weight.requires_grad = True
        # 内容分类词嵌入初始化 
        self.section_embed = torch.nn.Embedding(len(section_mappings), self.section_dim)
        self.section_embed.weight.requires_grad = True
        # 初始化解码器的开始状态。Parameter是Tensor的一种.自动注册为模型的参数，追踪梯度
        self.decoder_start = nn.Parameter(torch.FloatTensor(hidden_size*2).normal_())
        # mlp将rnn输出的隐藏状态映射到输出
        # 句子排序
        self.sent2out = self.sent_mlp(hidden_size*2, mlp_size, 0.3)
        # 摘要结构生成
        self.discourse2out = self.discourse_mlp(hidden_size*2, mlp_size, 0.3)

        self.sent_layer = nn.Linear(mlp_size, 1)
        self.discourse_layer = nn.Linear(mlp_size, 6)

        self.dropout_layer = nn.Dropout(p=0.3)
        # 注意力机制,输入，输出权重
        self.attn = nn.Linear(self.hidden_size*2*2, 1)
    # 输出分数用
    def sent_mlp(self, input, output, dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU(inplace=False))
        mlp.append(nn.Dropout(p=dropout, inplace=False))
        # nn.Sequential 是 PyTorch 中的一个容器，它可以接受一系列的神经网络层（如线性层、激活函数、池化层等），并按照顺序将它们连接起来，形成一个神经网络模型。
        return nn.Sequential(*mlp)
    # 输出结构用
    def discourse_mlp(self, input, output, dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU(inplace=False))
        mlp.append(nn.Dropout(p=dropout, inplace=False))
        return nn.Sequential(*mlp)

    def calc_context(self, decoder_state, encoder_outputs):
        """
        计算上下文
        """
        attn_weight = torch.cat((decoder_state.expand_as(encoder_outputs), encoder_outputs), dim=2)  # sent_output = [batch, sent_seq_len, hidden_size*2]

        attn_weight = F.softmax(self.attn(attn_weight), dim=1)

        attn_applied = torch.bmm(attn_weight.permute(0, 2, 1), encoder_outputs).squeeze(1)  

        return attn_applied

    def forward(self, inputs, doc_lengths, sentence_name, abstract_discourses, content_discourses, device):
        

        # 检查是否有nan
        has_nan=torch.isnan(inputs)
        nam_num = torch.sum(has_nan).item()
        if  nam_num> 0:
            print('inputs:',nam_num)
            print('inputs:',has_nan)
            # 找到包含NaN值的位置
            nan_indices = torch.nonzero(has_nan).squeeze()

            # 输出结果
            print("Indices of NaN values:", nan_indices)
            input('nan in inputs')


        # sent_discourse
        # 句子的分类标签嵌入
        discourse_embedded = self.discourse_embed(content_discourses)
        # print('inputs:',inputs.size())    # inputs = torch.Size([sent_seq_len, batch, hidden_size])
        # print('content_discourses:', content_discourses.size()) # content_discourses = torch.Size([sent_seq_len, batch])
        # print('discourse_embedded:', discourse_embedded.size()) # discourse_embedded = [sent_seq_len, batch, discourse_dim]

        # sent_section
        section_embedded = self.section_embed(sentence_name)
        # print('section_embedded:',section_embedded.size())    # inputs = torch.Size([sent_seq_len, batch, section_dim])

        # sent encoder
        # sent_in = torch.cat((inputs, section_embedded), dim=2) 
        # sent_in = torch.cat((inputs, discourse_embedded), dim=2) 
        # 将多个1*n 的张量拼接成一个1*m的张量
        sent_in = torch.cat((inputs, discourse_embedded, section_embedded), dim=2)  
        # print('sent_in:',sent_in.size())  # sent_in = torch.Size([sent_seq_len, batch, hidden_size+discourse_dim])
        # 把数据填充到一个固定长度
        packed_padded_doc_batch = pack_padded_sequence(sent_in, doc_lengths)
        # 双循环表征句子
        sent_output, sent_hidden = self.encoder_rnn_sent(packed_padded_doc_batch)
        # 解包成张量，方便使用
        sent_output, _ = pad_packed_sequence(sent_output) 
        # print('sent_output:', sent_output.size())   # sent_output = [sent_seq_len, batch, hidden_size*2]
        # print('sent_hidden:', sent_hidden.size())   # sent_hidden = [2, batch, hidden_size*2]
        # 过一次mlp，得出句子的得分
        sent_mlp_out = self.sent2out(sent_output)   # sent_mlp_out = [sent_seq_len, batch, mlp_size]
        # 线性层？？
        sent_out = self.sent_layer(sent_mlp_out)   
        # print('sent_out:', sent_out.size())  # sent_out = [sent_seq_len, batch, 1]

        # discourse decoder############################################################################################################
        # 批次大小
        batch_size = abstract_discourses.size(0)
        # 句子的数量
        sequence_size = abstract_discourses.size(1)

        # <start>
        # decoder_state = self.decoder_start.view(1, 1, -1).repeat(1, batch_size, 1) # decoder_state = [1, batch, hidden_size*2]
        ##???????????????
        decoder_state = torch.cat((torch.split(sent_hidden, 1, dim=0)), dim=2)     # decoder_state = [1, batch, hidden_size*2]
        # print('decoder_state:', decoder_state.size())  # decoder_state = [1, batch, hidden_size*2]
 
        # <GO>
        output = Variable(torch.ones((batch_size))).long().to(device)
        # print('output:', output)  # sent_out = [batch]

        actions = []
        logits = []
        for t in range(sequence_size):
            # print('t:',t)
            # 随机一个0-1之间的数，用来判断是否使用teacher_forcing
            random_ratio = random()
            # print('random_ratio:',random_ratio)
            # 如果随机数小于teacher_forcing_ratio，则不使用teacher_forcing
            use_teacher_forcing = False if random_ratio <= self.teacher_forcing_ratio else True
            # print('use_teacher_forcing:',use_teacher_forcing)

            if use_teacher_forcing:
                decoder_state_h = decoder_state
                # print('decoder_state_h:', decoder_state_h.size())    # decoder_state_h = [1, batch, hidden_size*2]
                # 整个语义空间的计算
                context = self.calc_context(decoder_state_h.permute(1, 0, 2), sent_output.permute(1, 0, 2))
                # print('context:', context.size())    # context = [batch, hidden_size*2]

                # discourse label embed
                discourse_embedded_input = self.discourse_embed(output)  # discourse_embedded_input = [batch, discourse_dim]
                discourse_embedded_input = self.dropout_layer(discourse_embedded_input)
                # print('discourse_embedded_input:', discourse_embedded_input.size()) 

                decoder_input_t = torch.cat([discourse_embedded_input, context], 1) # discourse_embedded_input = [batch, hidden_size*2 + discourse_dim]

                decoder_output_t, decoder_state = self.decoder_rnn(decoder_input_t.unsqueeze(0), decoder_state)
                # print('decoder_output_t:', decoder_output_t.size())   # decoder_output_t = [1, batch, hidden_size*2]
                # print('decoder_state:', decoder_state.size())         # decoder_state = [1, batch, hidden_size*2]

                discourse_mlp_out = self.discourse2out(decoder_output_t)  
                # print('discourse_mlp_out:', discourse_mlp_out.size())     # discourse_mlp_out = [1, batch, mlp_size]

                discourse_out = self.discourse_layer(discourse_mlp_out.squeeze(0))  
                # print('discourse_out:', discourse_out.size())    # discourse_out = [batch, 6]

                logit = F.log_softmax(discourse_out, dim=1)
                # print('logit:', logit.size())  # torch.Size([128, 6])

                output = abstract_discourses[:, t]
                # print('output:',output.size())

                logits.append(logit) 
                actions.append(output)

            else:
                decoder_state_h = decoder_state
                # print('decoder_state_h:', decoder_state_h.size())    # decoder_state_h = [1, batch, hidden_size*2]

                context = self.calc_context(decoder_state_h.permute(1, 0, 2), sent_output.permute(1, 0, 2))
                # print('context:', context.size())    # context = [batch, hidden_size*2]

                # discourse label embed
                discourse_embedded_input = self.discourse_embed(output)  # discourse_embedded_input = [batch, discourse_dim]
                discourse_embedded_input = self.dropout_layer(discourse_embedded_input)
                # print('discourse_embedded_input:', discourse_embedded_input.size()) 

                decoder_input_t = torch.cat([discourse_embedded_input, context], 1) # discourse_embedded_input = [batch, hidden_size*2 + discourse_dim]

                decoder_output_t, decoder_state = self.decoder_rnn(decoder_input_t.unsqueeze(0), decoder_state)
                # print('decoder_output_t:', decoder_output_t.size())   # decoder_output_t = [1, batch, hidden_size*2]
                # print('decoder_state:', decoder_state.size())         # decoder_state = [1, batch, hidden_size*2]

                discourse_mlp_out = self.discourse2out(decoder_output_t)  
                # print('discourse_mlp_out:', discourse_mlp_out.size())     # discourse_mlp_out = [1, batch, mlp_size]

                discourse_out = self.discourse_layer(discourse_mlp_out.squeeze(0))  
                # print('discourse_out:', discourse_out.size())    # discourse_out = [batch, 6]

                logit = F.log_softmax(discourse_out, dim=1)
                # print('logit:', logit.size())  # torch.Size([128, 6])

                output = torch.argmax(logit, dim=1).detach()
                # print('output:',output.size())

                logits.append(logit) 
                actions.append(output)
        # # 为nan的位置是True
        # if torch.sum(torch.isnan(sent_out)).item() > 0:
        #     print('sent_out:',torch.isnan(sent_out))
        #     input('nan in sent_out')
        

        return sent_out, logits, actions

    def predict(self, content_discourse_dir, score_batch, discourse_preds, ids, input_lengths, length_limit, filenames, hyp_path):

        # get discourse pattern 
        discourse_preds = discourse_preds.to('cpu').numpy() 
        # print('discourse_preds:', discourse_preds)

        all_discourse_preds = []
        for preds in discourse_preds:
            preds2label = [reverse_discourse_mappings[pred] for pred in preds if pred != 0]
            all_discourse_preds.append(preds2label)

        # print('discourse_preds:', len(discourse_preds))
        # print('all_discourse_preds:', len(all_discourse_preds))

        # get predicted summary
        summaryfile_batch = []
        all_ids = []
        for i in range(len(input_lengths)):

            # get input data
            fn = filenames[i]
            with fn.open() as of:
                data = json.load(of)
            inputs = data['inputs']
            idx = data['id']

            # get discourse label
            discourse_file = Path(content_discourse_dir)/"{}.json".format(idx)
            with discourse_file.open() as of:
                discourse_data = json.load(of)

            # get sorted index of predicted scores
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _, x in sorted(zip(scores, list(range(input_lengths[i]))), reverse=True)]
            # print('sorted_linenum:',sorted_linenum)
            
            # group_by_label
            sorted_sent_labels = [discourse_data[index_sent_id] for index_sent_id in sorted_linenum]
            # print('sorted_sent_labels:',sorted_sent_labels)

            label_cluster_dict = get_clusters(sorted_linenum, sorted_sent_labels)
            # print('label_cluster_dict:',label_cluster_dict.keys())

            # get predicted summary
            summary = []
            selected_ids = []

            content_plan = all_discourse_preds[i]
            # print('content_plan:',content_plan)

            if len(sorted_linenum) < len(content_plan):
                for sent_id in sorted_linenum:
                    summary.append(inputs[sent_id]['text'])
            else:
                num_out_of_plan = 0
                for plan in content_plan:

                    if plan in label_cluster_dict.keys():

                        if label_cluster_dict[plan] != []:
                            sentence_index_candidates = label_cluster_dict[plan]
                            # print('sentence_index_candidates:',sentence_index_candidates)

                            sent_id = sentence_index_candidates[0]
                            # print('sent_id:',sent_id)

                            selected_ids.append(sent_id)
                            summary.append(inputs[sent_id]['text'])

                            # update the dict 
                            try:
                                label_cluster_dict[plan] = sentence_index_candidates[1:]
                                # print('label_cluster_dict[plan]:',label_cluster_dict[plan])
                            except:
                                label_cluster_dict[plan] = []

                        else:
                            sent_id = sorted_linenum[0]
                            selected_ids.append(sent_id)
                            summary.append(inputs[sent_id]['text'])

                            num_out_of_plan += 1
                            sorted_linenum = sorted_linenum[1:]

                    else:
                        sent_id = sorted_linenum[0]
                        selected_ids.append(sent_id)
                        summary.append(inputs[sent_id]['text'])

                        num_out_of_plan += 1
                        sorted_linenum = sorted_linenum[1:]

            # save predicted summary
            summary = '\n'.join(summary)
            fname = hyp_path + ids[i] + '.txt'
            of = open(fname, 'w')
            of.write(summary)

            summaryfile_batch.append(fname)
            all_ids.append(selected_ids)

        # print('num_out_of_plan:',num_out_of_plan)

        return summaryfile_batch, all_ids


# Ext_embedding
class Ext_Emb_summ(nn.Module):
    def __init__(self, input_size, hidden_size, discourse_dim, section_dim, mlp_size, cell_type='gru'):
        super(Ext_Emb_summ, self).__init__()
        self.hidden_size = hidden_size
        self.discourse_dim = discourse_dim
        self.section_dim = section_dim
        self.cell = cell_type

        if self.cell == 'gru':
            #                   拼接后的数据，三个张量维度相加                   隐藏层的维度  一层            双向
            self.rnn = nn.GRU(input_size+self.discourse_dim+self.section_dim, hidden_size, num_layers=1, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True)
        # 词嵌入                                    词嵌入的词数量             词嵌入的维度
        self.discourse_embed = torch.nn.Embedding(len(discourse_mappings), self.discourse_dim)
        # 文本词嵌入是训练好的glove，这里的标签词嵌入需要自己训练
        self.discourse_embed.weight.requires_grad = True

        self.section_embed = torch.nn.Embedding(len(section_mappings), self.section_dim)
        self.section_embed.weight.requires_grad = True

        # 编码器的输出，mlp            句子使用双向循环拼接，所以是hidden_size*2
        self.sent2out = self.sent_mlp(hidden_size*2, mlp_size, 0.3)
        self.sent_layer = nn.Linear(mlp_size, 1)

    def sent_mlp(self, input, output, dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU(inplace=False))
        mlp.append(nn.Dropout(p=dropout, inplace=False))
        return nn.Sequential(*mlp)
        """
        nn.Sequential 是 PyTorch 中的一个容器，它可以接受一系列的神经网络层（如线性层、激活函数、池化层等），并按照顺序将它们连接起来，形成一个神经网络模型。
        """

    def forward(self, inputs, doc_lengths, sentence_name, content_discourses, device):

        # sent_discourse
        discourse_embedded = self.discourse_embed(content_discourses)
        # print('inputs:',inputs.size())    # inputs = torch.Size([sent_seq_len, batch, hidden_size])
        # print('content_discourses:', content_discourses.size()) # content_discourses = torch.Size([sent_seq_len, batch])
        # print('discourse_embedded:', discourse_embedded.size()) # discourse_embedded = [sent_seq_len, batch, discourse_dim]

        # sent_section
        section_embedded = self.section_embed(sentence_name)
        # print('section_embedded:',section_embedded.size())    # inputs = torch.Size([sent_seq_len, batch, section_dim])

        # sent encoder
        sent_in = torch.cat((inputs, discourse_embedded, section_embedded), dim=2)  
        # print('sent_in:',sent_in.size())  # sent_in = torch.Size([sent_seq_len, batch, hidden_size+discourse_dim])

        packed_padded_doc_batch = pack_padded_sequence(sent_in, doc_lengths)
        sent_output, sent_hidden = self.rnn(packed_padded_doc_batch)
        sent_output, _ = pad_packed_sequence(sent_output) 
        # print('sent_output:', sent_output.size())   # sent_output = [sent_seq_len, batch, hidden_size*2]
        # print('sent_hidden:', sent_hidden.size())   # sent_hidden = [2, batch, hidden_size*2]

        sent_mlp_out = self.sent2out(sent_output)   # sent_mlp_out = [sent_seq_len, batch, mlp_size]
        sent_out = self.sent_layer(sent_mlp_out)   
        # print('sent_out:', sent_out.size())  # sent_out = [sent_seq_len, batch, 1]

        return sent_out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames, hyp_path):
        #score_batch = [batch, seq_len]
        summaryfile_batch = []
        all_ids = []

        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _, x in sorted(zip(scores, list(range(input_lengths[i]))), reverse=True)]
            fn = filenames[i]

            with fn.open() as of:
                inputs = json.load(of)['inputs']

            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc += inputs[j]['word_count']
                if wc >= length_limit:
                    break
            summary = '\n'.join(summary)

            fname = hyp_path + ids[i] + '.txt'
            of = open(fname,'w')
            of.write(summary)

            all_ids.append(selected_ids)
            summaryfile_batch.append(fname)

        return summaryfile_batch, all_ids