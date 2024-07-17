## Data
The original data: https://github.com/armancohan/long-summarization.<br/>
Our preprocessed data: <a href = 'https://drive.google.com/file/d/1rJeEYJmpqhNOgOIfB3B2yxraL2WEsV4g/view?usp=sharing'>arXiv</a>,<a href='https://drive.google.com/file/d/1v4quWNb4ujVrzLhdDsAyyTiHKdrC4Dg8/view?usp=sharing'>Pubmed</a> <br/>

```
{
"id": "PMC88811", 
"inputs": 
	[
		{"text": "as a consequence of the availability of whole - genome expression methodologies , regulation of gene expression is at the core of current post - genomic studies .", 
		"tokens": ["as", "a", "consequence", "of", "the", "availability", "of", "whole", "-", "genome", "expression", "methodologies", ",", "regulation", "of", "gene", "expression", "is", "at", "the", "core", "of", "current", "post", "-", "genomic", "studies", "."], 
		"sentence_id": 1, 
		"word_count": 28}, 
		{}, ...
	]

"section_names": ["background", "description of datasets", "levels of analysis", "dyad analysis", "results", "pattern discovery", "sequence alignment", 	"detection of new members of regulons by pattern matching", "predictions", "comparison of results with recently examined members of the lexa regulon", "conclusions", "additional data files", "supplementary material", "acknowledgements", "figures and tables"],

"section_lengths": [53, 16, 15, 8, 85, 16, 20, 26, 15, 7, 22, 1, 1, 1, 56]
}

{"id": "PMC88811", "labels": [0, 0,....]}

note that: sum(section_lengths) == last sentence_id
```


> 在安装pytorch的基础上，使用`pip install -r requirements.txt` 安装其它库

## Train
To train the model, just type 

``` 
++++++++++++++++++++++ ext_summ

python main.py  --train_input inputs/train/ --train_label labels-greedy/train/ --train_abstract_discourse abstract-discourses/train/ --train_content_discourse content-discourses/train/ --val_input inputs/val/ --val_label labels-greedy/val/ --val_abstract_discourse abstract-discourses/val/ --refpath human-abstracts/val/ --gloveDir ./pretrained_embeddings --val_content_discourse content-discourses/val/ --length_limit 220 --batchsize 48 --dataset arxiv --device 0 --model ext_summ --mode validate --runtime 0 


python main.py  --train_input inputs/train/ --train_label labels-greedy/train/ --train_abstract_discourse abstract-discourses/train/ --train_content_discourse content-discourses/train/ --val_input inputs/val/ --val_label labels-greedy/val/ --val_abstract_discourse abstract-discourses/val/ --refpath human-abstracts/val/ --gloveDir ./pretrained_embeddings --val_content_discourse content-discourses/val/ --length_limit 200 --batchsize 48 --dataset pubmed --device 0 --model ext_summ --mode validate --runtime 0

++++++++++++++++++++++ ext_emb_summ

python main.py  --train_input inputs/train/ --train_label labels-greedy/train/ --train_abstract_discourse abstract-discourses/train/ --train_content_discourse content-discourses/train/ --val_input inputs/val/ --val_label labels-greedy/val/ --val_abstract_discourse abstract-discourses/val/ --refpath human-abstracts/val/ --gloveDir ./pretrained_embeddings --val_content_discourse content-discourses/val/ --length_limit 220 --batchsize 48 --dataset arxiv --device 0 --model ext_emb_summ --mode validate --runtime 0 


python main.py  --train_input inputs/train/ --train_label labels-greedy/train/ --train_abstract_discourse abstract-discourses/train/ --train_content_discourse content-discourses/train/ --val_input inputs/val/ --val_label labels-greedy/val/ --val_abstract_discourse abstract-discourses/val/ --refpath human-abstracts/val/ --gloveDir ./pretrained_embeddings --val_content_discourse content-discourses/val/ --length_limit 200 --batchsize 48 --dataset pubmed --device 0 --model ext_emb_summ --mode validate --runtime 0

++++++++++++++++++++++ multi_sent_discourse_summ

python main.py  --train_input inputs/train/ --train_label labels-greedy/train/ --train_abstract_discourse abstract-discourses/train/ --train_content_discourse content-discourses/train/ --val_input inputs/val/ --val_label labels-greedy/val/ --val_abstract_discourse abstract-discourses/val/ --refpath human-abstracts/val/ --gloveDir ./pretrained_embeddings --val_content_discourse content-discourses/val/ --length_limit 220 --batchsize 64 --dataset arxiv --device 0 --model multi_sent_discourse_summ --mode validate --content_size 10 --teacher_forcing_ratio 1.5 --discourse_dim 32 --section_dim 32 --runtime both-10-64-32-32-1.5-1 


python main.py  --train_input inputs/train/ --train_label labels-greedy/train/ --train_abstract_discourse abstract-discourses/train/ --train_content_discourse content-discourses/train/ --val_input inputs/val/ --val_label labels-greedy/val/ --val_abstract_discourse abstract-discourses/val/ --refpath human-abstracts/val/ --gloveDir ./pretrained_embeddings --val_content_discourse content-discourses/val/ --length_limit 200 --batchsize 64 --dataset pubmed --device 0 --model multi_sent_discourse_summ --mode validate --content_size 8 --teacher_forcing_ratio 1.5 --discourse_dim 32 --section_dim 32 --runtime both-8-64-32-32-1.5-1




```

## Test
To test the model, just type 
```
++++++++++++++++++++++ ext_summ

python test.py --test_input inputs/test/ --test_label labels/test/ --test_abstract_discourse abstract-discourses/test/ --refpath human-abstracts/test/ --gloveDir ./pretrained_embeddings --test_content_discourse content-discourses/test/ --length_limit 200 --dataset arxiv --device 0 --model ext_summ --model_path pretrained_models/ --mode test  --epoch 34 --runtime 0

python test.py --test_input inputs/test/ --test_label labels/test/ --test_abstract_discourse abstract-discourses/test/ --refpath human-abstracts/test/ --gloveDir ./pretrained_embeddings --test_content_discourse content-discourses/test/ --length_limit 200 --dataset pubmed --device 0 --model ext_summ --model_path pretrained_models/ --mode test  --epoch 40 --runtime 0

++++++++++++++++++++++ ext_emb_summ

python test.py --test_input inputs/test/ --test_label labels/test/ --test_abstract_discourse abstract-discourses/test/ --refpath human-abstracts/test/ --gloveDir ./pretrained_embeddings --test_content_discourse content-discourses/test/ --length_limit 200 --dataset arxiv --device 0 --model ext_emb_summ --model_path pretrained_models/ --mode test  --epoch 20 --runtime 0

python test.py --test_input inputs/test/ --test_label labels/test/ --test_abstract_discourse abstract-discourses/test/ --refpath human-abstracts/test/ --gloveDir ./pretrained_embeddings --test_content_discourse content-discourses/test/ --length_limit 200 --dataset pubmed --device 0 --model ext_emb_summ --model_path pretrained_models/ --mode test  --epoch 15 --runtime 0

++++++++++++++++++++++ sent_sect_discourse_summ

python test.py --test_input inputs/test/ --test_label labels/test/ --test_abstract_discourse abstract-discourses/test/ --refpath human-abstracts/test/ --gloveDir ./pretrained_embeddings --test_content_discourse content-discourses/test/ --length_limit 200 --dataset arxiv --device 0 --model multi_sent_discourse_summ --model_path pretrained_models/ --mode test --content_size 10 --teacher_forcing_ratio 1.5 --discourse_dim 48 --section_dim 48 --epoch 8 --runtime both-10-48-48-1.5

python test.py --test_input inputs/test/ --test_label labels/test/ --test_abstract_discourse abstract-discourses/test/ --refpath human-abstracts/test/ --gloveDir ./pretrained_embeddings --test_content_discourse content-discourses/test/ --length_limit 200 --dataset pubmed --device 0 --model multi_sent_discourse_summ --model_path pretrained_models/ --mode test --content_size 10 --teacher_forcing_ratio 1.5 --discourse_dim 32 --section_dim 32 --epoch 40 --runtime both-9-64-32-32-1.5-1

```




