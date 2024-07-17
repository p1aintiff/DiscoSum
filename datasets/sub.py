from pathlib import Path
import json
import os
from tqdm import tqdm
import shutil
import random

class CopyOne:
    def __init__(self,file_name,sub,dataset_path='./datasets/'):
        self.file_name = file_name
        self.input = os.path.join(dataset_path,"arxiv/inputs",sub,file_name)
        self.label = os.path.join(dataset_path,"arxiv/labels",sub,file_name)
        self.abstract_discourses = os.path.join(dataset_path,"arxiv/abstract-discourses",sub,file_name)
        self.content_discourses = os.path.join(dataset_path,"arxiv/content-discourses",sub,file_name)
        self.human_abstracts = os.path.join(dataset_path,"arxiv/human-abstracts",sub,file_name.replace(".json",".txt"))
        self.section_labels = os.path.join(dataset_path,"arxiv/section-labels",sub,file_name)
        

        self.input_sentence = 0
        self.delete_tag = False
        
        # 检查可用
        self.check()
        # 复制
        if not self.delete_tag:
            self.copy_target()
            


    def check(self):
        # 检查input内容存在
        if not self.exist_input():
            tqdm.write("input内容不存在")
            self.delete_tag = True
            return

        # 检查各种标签文件是否存在
        if not self.exist_labels():
            tqdm.write("标签文件不存在")
            self.delete_tag = True
            return

        # 检查labels数量
        if not self.num_labels():
            tqdm.write("labels数量错误")
            self.delete_tag = True
            return

        # 检查content-discourses数量
        if not self.num_content_discourses():
            tqdm.write("content-discourses数量错误")
            self.delete_tag = True
            return
        
    def copy_target(self):
        
        shutil.copyfile(self.input, self.input.replace('dataset','preset'))
        shutil.copyfile(self.label, self.label.replace('dataset','preset'))
        shutil.copyfile(self.abstract_discourses, self.abstract_discourses.replace('dataset','preset'))
        shutil.copyfile(self.content_discourses, self.content_discourses.replace('dataset','preset'))
        shutil.copyfile(self.human_abstracts, self.human_abstracts.replace('dataset','preset'))
        shutil.copyfile(self.section_labels, self.section_labels.replace('dataset','preset'))
        
    def exist_input(self):
        try:
            with open(self.input,"r") as f:
                data = json.load(f)
            self.input_sentence = sum(data["section_lengths"])
            ## 还要检查是否有空句子
            for idx, sent in enumerate(data['inputs']):
                if sent['word_count'] <1:
                    return False
            return True
        except:
            tqdm.write(f"{self.file_name}文件内容异常")
            return False
    

    def exist_labels(self):
        """
        各种label的存在性
        """
        if os.path.exists(self.abstract_discourses) and os.path.exists(self.content_discourses) and os.path.exists(self.human_abstracts) and os.path.exists(self.section_labels):
            return True
        else:
            return False
    
    def num_labels(self):
        """
        labels的数量
        """
        with open(self.label,"r") as f:
            data = json.load(f)
        
        if len(data["labels"])==self.input_sentence:
            return True
        else:
            return False
    
    def num_content_discourses(self):
        with open(self.content_discourses,"r") as f:
            data = json.load(f)
        if len(data)==self.input_sentence:
            return True
        else:
            return False


class datasets:
    def __init__(self):
        self.subs = ["test"]
        

    def get_input_list(self):
        for sub in self.subs:
            file_names=[]
            file_names.extend(os.listdir(f"./datasets/arxiv/inputs/{sub}"))
            print(f"{sub}共有{len(file_names)}个文件")
            
            
            # 这里开始抽取
            
            prelist = random.sample(file_names,5)
            for file_name in prelist:
                CopyOne(file_name,sub)
            
            
            # for file_name in tqdm(file_names, desc=f"{sub}数据检查"):
            #     RemoveOne(file_name,sub)



if __name__ == '__main__':
    datasets = datasets()
    datasets.get_input_list()
