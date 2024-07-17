from pathlib import Path
import json
import os
from tqdm import tqdm
from threading import Thread

class DataOne:
    def __init__(self,file_name,sub):
        self.file_name = file_name
        self.input = os.path.join("./datasets/pubmed/inputs",sub,file_name)
        self.label = os.path.join("./datasets/pubmed/labels",sub,file_name)
        self.abstract_discourses = os.path.join("./datasets/pubmed/abstract-discourses",sub,file_name)
        self.content_discourses = os.path.join("./datasets/pubmed/content-discourses",sub,file_name)
        self.human_abstracts = os.path.join("./datasets/pubmed/human-abstracts",sub,file_name.replace(".json",".txt"))
        self.section_labels = os.path.join("./datasets/pubmed/section-labels",sub,file_name)

        self.input_sentence = 0
        self.delete_tag = False

        self.check()
        if self.delete_tag:
            self.remove()


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

    def remove(self):
        """
        删除验证失败的数据
        """
        if self.delete_tag:
            tqdm.write(f"删除文件{self.file_name}")
            try:
                os.remove(self.input)
            except FileNotFoundError:
                pass
            try:
                os.remove(self.label)
            except FileNotFoundError:
                pass
            try:
                os.remove(self.abstract_discourses)
            except FileNotFoundError:
                pass
            try:
                os.remove(self.content_discourses)
            except FileNotFoundError:
                pass
            try:
                os.remove(self.human_abstracts)
            except FileNotFoundError:
                pass
            try:
                os.remove(self.section_labels)
            except FileNotFoundError:
                pass

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
        self.subs = ["train", "val", "test"]

    def process_sub(self, sub):
        """
        处理单个子目录的数据
        """
        file_names = os.listdir(f"./datasets/pubmed/inputs/{sub}")
        threads = []

        for file_name in tqdm(file_names, desc=f"{sub}数据检查"):
            # 创建线程并启动
            thread = Thread(target=self.process_file, args=(file_name, sub))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

    def process_file(self, file_name, sub):
        """
        线程工作函数，处理单个文件
        """
        data_one = DataOne(file_name, sub)

    def get_input_list(self):
        for sub in self.subs:
            print(f"开始处理 {sub} 数据")
            self.process_sub(sub)

if __name__ == '__main__':
    datasets = datasets()
    datasets.get_input_list()
