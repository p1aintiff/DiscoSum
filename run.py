from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *
from seqeval.metrics import f1_score, accuracy_score
import time



discourse_mappings = {'PAD_label': 0, 'BAC_label': 1, 'OBJ_label': 2, 'APP_label': 3, 'OUT_label': 4, 'OTH_label': 5}
reverse_discourse_mappings = {0:'PAD_label', 1:'BAC_label', 2:'OBJ_label', 3:'APP_label', 4:'OUT_label', 5:'OTH_label'}


def train_seq2seq(args, train_dataloader, model, optimizer, pos_weight, device, model_name, teacher_forcing,epoch_idx=1,writer=None):
    model.train()

    total_loss = 0
    total_data = 0
    for batch_idx, data in enumerate(train_dataloader):
    #     if batch_idx>83:
    #         print(f"file ids\n {data['id']}")
    #     print(f"train batch {batch_idx}",end='  ')

        start_time = time.time()
        len_train_dataloader = len(train_dataloader)
        l, num_data = train_seq2seq_batch(args, data, model, optimizer, pos_weight, device, model_name, teacher_forcing,batch_idx,epoch_idx,len_train_dataloader,writer)
        total_loss += l
        total_data += num_data
        # print("total_loss:",total_loss)
        # print("total_data:",total_data)

        # 记录当前batch的loss
        writer.add_scalar('train/batch_avg_loss', total_loss/float(total_data), batch_idx+epoch_idx*len(train_dataloader))

        # 记录当前batch的时间
        end_time = time.time()
        batch_time = end_time - start_time
        writer.add_scalar('train/Batch Time', batch_time, batch_idx+epoch_idx*len(train_dataloader))

        if batch_idx+epoch_idx*len(train_dataloader)%10==0:
            print('Batch %d, Loss: %f'%(batch_idx,total_loss/float(total_data)))

    return total_loss/float(total_data)


def train_seq2seq_batch(args, data_batch, model, optimizer, pos_weight, device, model_name, teacher_forcing=False,batch_idx=1,epoch_idx=1,len_train_dataloader=1,writer=None):
    document = data_batch['document']
    label = data_batch['labels']
    name = data_batch['names']
    input_length = data_batch['input_length']
    doc_lengths = data_batch['doc_lengths']
    abstract_discourses = data_batch['abstract_discourses']
    content_discourses = data_batch['content_discourses']

    total_data = torch.sum(input_length)
    content_size = abstract_discourses.size(1)
    # if content_size > args.content_size:
    #     content_size = args.content_size

    if torch.cuda.is_available():
        document = document.to(device)
        label = label.to(device)
        name = name.to(device)
        input_length = input_length.to(device)
        abstract_discourses = abstract_discourses.to(device)
        content_discourses = content_discourses.to(device)

    # print('document:', document.size())               # torch.Size([795, 128, 300]) [sent_seq_len, batch, hidden_size]
    # print('content_discourses:', content_discourses.size())   
    # print('label:', label.size())                     # torch.Size([795, 128, 1])   [sent_seq_len, batch, 1]
    # print('input_length:', input_length.size())       # torch.Size([128])           [batch]
    # print('abstract_discourses:', abstract_discourses.size())           # torch.Size([128, 10])       [batch, max_sent_seq_len]

    if model_name == 'ext_summ':
        # print('model_name:',model_name)
        sent_out = model(document, doc_lengths, device)

        mask_sent = label.gt(-1).float()  # >-1 ==1; <-1 == 0
        loss = F.binary_cross_entropy_with_logits(sent_out, label, weight=mask_sent, reduction='sum', pos_weight=pos_weight)

        print(f'loss:{loss.data:4f}')

        model.zero_grad()
        loss.backward()
        optimizer.step()

        l = loss.data

        del document,label,input_length,loss,sent_out
        torch.cuda.empty_cache()

        return l, total_data

    if model_name == 'ext_emb_summ':
        # print('model_name:',model_name)
        sent_out = model(document, doc_lengths, name, content_discourses, device)

        mask_sent = label.gt(-1).float()  # >-1 ==1; <-1 == 0
        loss = F.binary_cross_entropy_with_logits(sent_out, label, weight=mask_sent, reduction='sum', pos_weight=pos_weight)

        print(f'loss:{loss.data:4f}')

        model.zero_grad()
        loss.backward()
        optimizer.step()

        l = loss.data

        del document,label,input_length,loss,sent_out
        torch.cuda.empty_cache()

        return l, total_data

    if model_name == 'multi_sent_discourse_summ':
        # print('model_name:',model_name)

        sent_out, logits, actions = model(document, doc_lengths, name, abstract_discourses, content_discourses, device)

        # sentence loss
        mask_sent = label.gt(-1).float()  # >-1 ==1; <-1 == 0; padding_value=-1
        loss_sent = F.binary_cross_entropy_with_logits(sent_out, label, weight=mask_sent, reduction='sum', pos_weight=pos_weight)
        # print('mask_sent:', mask_sent.size())  # torch.Size([sent_seq_len, batch, 1])

        # loss_discourse
        loss_discourse = 0

        loss_func = nn.NLLLoss()
        for t in range(content_size):     
            # print('logits[t]:', logits[t].size())                #  torch.Size([128, 5])
            # print('abstract_discourses[:, t]:', abstract_discourses[:, t].size())  #  torch.Size([128])
            loss_discourse = loss_discourse + loss_func(logits[t], abstract_discourses[:, t])

        loss = loss_sent + 1.0*loss_discourse
        print(f'lambda:{1.0}, loss:{loss.data:4f}, loss_sent:{loss_sent.data:4f}, loss_discourse:{1.0*loss_discourse.data:4f}')
        writer.add_scalar('train/loss_sent', loss_sent.data, batch_idx+epoch_idx*len_train_dataloader)
        writer.add_scalar('train/loss_discourse', loss_discourse.data, batch_idx+epoch_idx*len_train_dataloader)
        model.zero_grad()
        loss.backward()


        
        # clip 
        # 梯度修剪，防止梯度爆炸
        # 设置随机噪声为1e-3，梯度裁剪为20
        gradient_noise_and_clip(model.parameters(), device, noise_stddev=1e-3, max_clip=20)

        # 记录当前batch的梯度
        # total_grad_norm = 0.0
        # for param in model.parameters():
        #     if param.grad is not None:
        #         total_grad_norm += param.grad.data.norm(2).item() ** 2
        # total_grad_norm = total_grad_norm ** 0.5
        # writer.add_scalar('train/Gradient Norm', total_grad_norm, batch_idx+epoch_idx*len_train_dataloader)
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_scalar('train/'+name, param.grad.norm(2).item(), batch_idx+epoch_idx*len_train_dataloader)
            else:
                writer.add_scalar('train/'+name, 0, batch_idx+epoch_idx*len_train_dataloader)
        
        # 记录当前batch的学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, batch_idx+epoch_idx*len_train_dataloader)
        optimizer.step()

        l = loss.data

        del document,label,input_length,abstract_discourses,loss,sent_out,logits,actions,loss_sent,loss_discourse
        torch.cuda.empty_cache()

        return l, total_data

    else:
        pass


def eval_seq2seq(args, val_dataloader, content_discourse_dir, model, hyp_path, length_limit, pos_weight, device, model_name):
    model.eval()
    print('eval')

    total_loss = 0
    total_data = 0
    total_correct = 0

    summ_path = []
    ref_path = []

    all_ids = []

    all_discourse_preds = []
    all_discourse_golds = []
    idxes = []

    sigmoid = torch.nn.Sigmoid()

    if model_name == 'ext_summ':
        for i, data in enumerate(val_dataloader):
            summaryfiles, referencefiles, loss, num_data, select_ids = eval_seq2seq_batch(args, content_discourse_dir, sigmoid, data, model, hyp_path, length_limit, pos_weight, device, model_name)        
            
            summ_path.extend(summaryfiles)
            ref_path.extend(referencefiles)
            all_ids.extend(select_ids)

            total_loss += loss
            total_data += num_data

            del data
            del loss

        rouge_result = get_rouge(args, summ_path, ref_path)

        rouge_1_f_score = rouge_result["rouge_1_f_score"] * 100
        rouge_2_f_score = rouge_result["rouge_2_f_score"] * 100
        rouge_l_f_score = rouge_result["rouge_l_f_score"] * 100

        rouge_f_score = round((rouge_1_f_score+rouge_2_f_score+rouge_l_f_score)/3,2)

        with open('./results/' + args.dataset + '_' + args.model + '_' + args.mode + '_sent__rouge_results.txt', 'w') as f:
            f.write('rouge_1_f_score:' + str(rouge_1_f_score) +'\n')
            f.write('rouge_2_f_score:' + str(rouge_2_f_score) +'\n')
            f.write('rouge_l_f_score:' + str(rouge_l_f_score) +'\n')

        return rouge_f_score, total_loss/float(total_data)

    if model_name == 'ext_emb_summ':
        for i, data in enumerate(val_dataloader):
            summaryfiles, referencefiles, loss, num_data, select_ids = eval_seq2seq_batch(args, content_discourse_dir, sigmoid, data, model, hyp_path, length_limit, pos_weight, device, model_name)        
            
            summ_path.extend(summaryfiles)
            ref_path.extend(referencefiles)
            all_ids.extend(select_ids)

            total_loss += loss
            total_data += num_data

            del data
            del loss

        rouge_result = get_rouge(args, summ_path, ref_path)

        rouge_1_f_score = rouge_result["rouge_1_f_score"] * 100
        rouge_2_f_score = rouge_result["rouge_2_f_score"] * 100
        rouge_l_f_score = rouge_result["rouge_l_f_score"] * 100

        rouge_f_score = round((rouge_1_f_score+rouge_2_f_score+rouge_l_f_score)/3,2)

        with open('results/' + args.dataset + '_' + args.model + '_' + args.mode + '_sent__rouge_results.txt', 'w') as f:
            f.write('rouge_1_f_score:' + str(rouge_1_f_score) +'\n')
            f.write('rouge_2_f_score:' + str(rouge_2_f_score) +'\n')
            f.write('rouge_l_f_score:' + str(rouge_l_f_score) +'\n')

        return rouge_f_score, total_loss/float(total_data)
    ######################    
    if model_name == 'multi_sent_discourse_summ':
        for ii, data in enumerate(val_dataloader):
            summaryfiles, referencefiles, loss, num_data, select_ids, discourse_preds = eval_seq2seq_batch(args, content_discourse_dir, sigmoid, data, model, hyp_path, length_limit, pos_weight, device, model_name)        
            
            summ_path.extend(summaryfiles)
            ref_path.extend(referencefiles)
            all_ids.extend(select_ids)

            discourse_golds = data['abstract_discourses']
            ids = data['id']

            discourse_preds = discourse_preds.to('cpu').numpy() 
            discourse_golds = discourse_golds.to('cpu').numpy()
            # print('ids:',ids[0]) 
            # print('discourse_preds:',discourse_preds[0])
            # print('discourse_golds:',discourse_golds[0])
            # print('discourse_preds_size:',discourse_preds.shape)
            # print('discourse_golds_size:',discourse_golds.shape)

            # for i, label in enumerate(discourse_golds):
            #     temp = []
            #     for j, m in enumerate(label):
            #         if discourse_golds[i][j] == 0: 
            #             all_discourse_golds.append(temp)
            #             break
            #         else:
            #             temp.append(reverse_discourse_mappings[discourse_golds[i][j]])

            # for i, label in enumerate(discourse_preds):
            #     temp = []
            #     for j, m in enumerate(label):
            #         if discourse_preds[i][j] == 0: 
            #             all_discourse_preds.append(temp)
            #             break
            #         else:
            #             temp.append(reverse_discourse_mappings[discourse_preds[i][j]])

            for i, label in enumerate(discourse_golds):
                temp = []
                for j, m in enumerate(label):
                    temp.append(reverse_discourse_mappings[discourse_golds[i][j]])
                all_discourse_golds.append(temp)

            for i, label in enumerate(discourse_preds):
                temp = []
                for j, m in enumerate(label):
                    temp.append(reverse_discourse_mappings[discourse_preds[i][j]])
                all_discourse_preds.append(temp)

            idxes.extend(ids)
            # print('idxes:',idxes)
            # print('all_discourse_golds:',all_discourse_golds[0])
            # print('all_discourse_preds:',all_discourse_preds[0])

            total_loss += loss
            total_data += num_data

            del data
            del loss

        # evaluate for rouge
        rouge_result = get_rouge(args, summ_path, ref_path)

        rouge_1_f_score = rouge_result["rouge_1_f_score"] * 100
        rouge_2_f_score = rouge_result["rouge_2_f_score"] * 100
        rouge_l_f_score = rouge_result["rouge_l_f_score"] * 100

        rouge_f_score = round((rouge_1_f_score+rouge_2_f_score+rouge_l_f_score)/3,2)

        with open('results/' + args.dataset + '_' + args.model + '_' + args.mode + '_sent_discourse_rouge_results.txt', 'w') as f:
            f.write('rouge_1_f_score:' + str(rouge_1_f_score) +'\n')
            f.write('rouge_2_f_score:' + str(rouge_2_f_score) +'\n')
            f.write('rouge_l_f_score:' + str(rouge_l_f_score) +'\n')


        # save for content plan
        # print('all_discourse_preds:',len(all_discourse_preds))
        # print('all_discourse_golds:',len(all_discourse_golds))

        with open('pred_content_plan/' + args.dataset + '_' + args.model + '_' + args.mode + '_content_plan_preds.txt', 'w') as f:
            for item in all_discourse_preds:
                f.write(' '.join(item) + '\n')

        with open('pred_content_plan/' + args.dataset + '_' + args.model + '_' + args.mode + '_content_plan_golds.txt', 'w') as f:
            for item in all_discourse_golds:
                f.write(' '.join(item) + '\n')

        # evaluate for discourse
        f1 = f1_score(all_discourse_golds, all_discourse_preds)
        accuracy = accuracy_score(all_discourse_golds, all_discourse_preds)
        print('f1:',f1)
        print('accuracy:',accuracy)

        with open('results/' + args.dataset + '_' + args.model + '_' + args.mode + '_sent_discourse_content_results.txt', 'w') as f:
            f.write('f1:' + str(f1) +'\n')
            f.write('accuracy:' + str(accuracy) +'\n')

        return rouge_f_score, total_loss/float(total_data)

    else:
        pass

def eval_seq2seq_batch(args, content_discourse_dir, sigmoid, data_batch, model, hyp_path, length_limit, pos_weight, device, model_name):

    document = data_batch['document']
    label = data_batch['labels']
    name = data_batch['names']
    input_length = data_batch['input_length']
    doc_lengths = data_batch['doc_lengths']
    abstract_discourses = data_batch['abstract_discourses']
    content_discourses = data_batch['content_discourses']

    total_data = torch.sum(input_length)

    content_size = abstract_discourses.size(1)
    if content_size > args.content_size:
        content_size = args.content_size

    if torch.cuda.is_available():
        document = document.to(device)
        label = label.to(device)
        name = name.to(device)
        input_length = input_length.to(device)
        abstract_discourses = abstract_discourses.to(device)
        content_discourses = content_discourses.to(device)

    reference = data_batch['refs']
    filenames = data_batch['filenames']
    ids = data_batch['id']

    if model_name == 'ext_summ':
        sent_out = model(document, doc_lengths, device)

        mask = label.gt(-1).float()
        loss = F.binary_cross_entropy_with_logits(sent_out, label, weight=mask, reduction='sum', pos_weight=pos_weight)
        print(f'loss:{loss.data:4f}')
        sent_out = sent_out.squeeze(-1)  # sent_out = [seq_len, batch, 1]
        scores = sigmoid(sent_out).data # scores = [seq_len, batch]
        scores = scores.permute(1, 0)
        # np.save(args.dataset+'_scores', scores.cpu().data.numpy())

        summaryfiles, all_ids = model.predict(scores, ids, input_length, length_limit, filenames, hyp_path)

        label = label.squeeze(-1)
        label = label.permute(1, 0)

        del document,label,input_length

        return summaryfiles, reference, loss.data, total_data, all_ids

    if model_name == 'ext_emb_summ':
        sent_out = model(document, doc_lengths, name, content_discourses, device)

        mask = label.gt(-1).float()
        loss = F.binary_cross_entropy_with_logits(sent_out, label, weight=mask, reduction='sum', pos_weight=pos_weight)
        
        sent_out = sent_out.squeeze(-1)  # sent_out = [seq_len, batch, 1]
        scores = sigmoid(sent_out).data # scores = [seq_len, batch]
        scores = scores.permute(1, 0)
        # np.save(args.dataset+'_scores', scores.cpu().data.numpy())

        summaryfiles, all_ids = model.predict(scores, ids, input_length, length_limit, filenames, hyp_path)

        label = label.squeeze(-1)
        label = label.permute(1, 0)

        del document,label,input_length

        return summaryfiles, reference, loss.data, total_data, all_ids

    if model_name == 'multi_sent_discourse_summ':
        sent_out, logits, actions = model(document, doc_lengths, name, abstract_discourses, content_discourses, device)

        # sentence loss
        mask = label.gt(-1).float()
        loss_sent = F.binary_cross_entropy_with_logits(sent_out, label, weight=mask, reduction='sum', pos_weight=pos_weight)
        
        # loss_discourse
        loss_discourse = 0

        loss_func = nn.NLLLoss()
        for t in range(content_size):     
            # print('logits[t]:', logits[t].size())                #  torch.Size([128, 5])
            # print('abstract_discourses[:, t]:', abstract_discourses[:, t].size())  #  torch.Size([128])
            loss_discourse = loss_discourse + loss_func(logits[t], abstract_discourses[:, t])

        # get predicted sentences
        sent_out = sent_out.squeeze(-1)      # out = [seq_len, batch, 1]
        sent_scores = sigmoid(sent_out).data # sent_scores = [batch, seq_len]
        sent_scores = sent_scores.permute(1, 0)
        # np.save(args.dataset+'_scores', sent_scores.cpu().data.numpy())

        # get predicted discourse
        discourse_preds = torch.cat(list(map(lambda x: x.unsqueeze(1), actions)), dim=1)

        # final output
        #  gold
        summaryfiles, all_ids = model.predict(content_discourse_dir, sent_scores, abstract_discourses, ids, input_length, length_limit, filenames, hyp_path)
        # summaryfiles, all_ids = model.predict(content_discourse_dir, sent_scores, discourse_preds, ids, input_length, length_limit, filenames, hyp_path)

        label = label.squeeze(-1)   # label = [seq_len, batch, 1]
        label = label.permute(1, 0) # label = [batch, seq_len]

        loss = loss_sent + loss_discourse

        del document,label,input_length,abstract_discourses,sent_scores,sent_out,logits,actions,loss_sent,loss_discourse

        return summaryfiles, reference, loss.data, total_data, all_ids, discourse_preds

    else:
        pass
