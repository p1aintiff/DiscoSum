import os
import sys
import codecs
import json
import random
import string
import shutil
import time
from multiprocessing import Pool
from pyrouge import Rouge155
import argparse
from nltk import sent_tokenize, word_tokenize


def load_txt(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def evaluate_rouge(data): 
    summaries, references,  = data
    rouge_dir = ''
    temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = os.path.join("temp", temp_dir)
    # print(temp_dir)
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    # print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    try:
        for i, (summary, candidates) in enumerate(zip(summaries, references)):
            summary_fn = '%i.txt' % i
            for j, candidate in enumerate(candidates):
                candidate_fn = '%i.%i.txt' % (i, j)
                with open(os.path.join(model_dir, candidate_fn), 'w') as f:
                    f.write('\n'.join(candidate))

            with open(os.path.join(system_dir, summary_fn), 'w') as f:
                f.write('\n'.join(summary))

        rouge = Rouge155(rouge_dir=rouge_dir)
        rouge.system_dir = system_dir
        rouge.model_dir = model_dir
        rouge.system_filename_pattern = '(\d+).txt'
        rouge.model_filename_pattern = '#ID#.\d+.txt'
        rouge_args = "-e {} -c 95 -r 1000 -n 2 -a".format(rouge.data_dir) 
        output = rouge.convert_and_evaluate(rouge_args=rouge_args)

        r = rouge.output_to_dict(output)
        # print(output)
    finally:
        shutil.rmtree(temp_dir)
    return r

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def test_rouge(candidates, references, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in candidates]
    references = [line.strip() for line in references]
    # print('references_before:',len(references))

    from nltk import sent_tokenize
    references = [[sent_tokenize(line)] for line in references]
    # references = [sent_tokenize(line) for line in references]
    candidates = [sent_tokenize(line) for line in candidates]
    # references = [[line.split("\t")] for line in references]
    # candidates = [line.split("\t") for line in candidates]

    # print(len(candidates))
    # print(len(references))
    assert len(candidates) == len(references)

    candidates_chunks = list(chunks(candidates, int(len(candidates)/num_processes)))
    references_chunks = list(chunks(references, int(len(references)/num_processes)))
    n_pool = len(candidates_chunks)

    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i]))
    pool = Pool(n_pool)

    results = pool.map(evaluate_rouge, arg_lst)

    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if(k not in final_results):
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)
    
    print(rouge_results_to_str(final_results))

    return final_results
    
def rouge_results_to_str(results_dict):
    return "ROUGE-F(1/2/3/l): {:.2f}  {:.2f}  {:.2f}".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
    # results_dict["rouge_1_recall"] * 100,
    # results_dict["rouge_2_recall"] * 100,
    # # results_dict["rouge_3_f_score"] * 100,
    # results_dict["rouge_l_recall"] * 100
    # ,results_dict["rouge_su*_f_score"] * 100
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', default='pubmed', type=str, help=['arxiv','pubmed'])
    parser.add_argument('-customiza_model', action='store_true')
    
    args = parser.parse_args()

    # read data and inference
    data_path_source = './outputs/' + args.data_name + '/' + 'baseline.test.summary'
    data_path_target = './datasets/' + args.data_name + '/' + 'test.target'

    summaries = load_txt(data_path_source)
    references = load_txt(data_path_target)
    
    assert len(references) == len(summaries)

    rouge_result = test_rouge(summaries, references, num_processes=8)