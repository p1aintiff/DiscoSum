from pathlib import Path
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
分别读取arxiv，pubmed训练集中，摘要的话语结构label
统计话语结构数量的频数
"""

BAC = "BAC_label"
OBJ = "OBJ_label"
APP = "APP_label"
OUT = "OUT_label"
OTH = "OTH_label"


# 数据集

class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # 各级路径
        self.dataset_path = Path(f"./datasets/{dataset_name}")
        self.inputs = os.path.join(self.dataset_path,"inputs")
        self.labels = os.path.join(self.dataset_path,"labels")
        self.abstract_discourses = os.path.join(self.dataset_path,"abstract-discourses")
        self.content_discourses = os.path.join(self.dataset_path,"content-discourses")
        self.human_abstracts = os.path.join(self.dataset_path,"human-abstracts")
        self.section_labels = os.path.join(self.dataset_path,"section-labels")
        
    
    def get_sub(self,sub):
        """
        获取数据集子集中所有 摘要话语结构 文件路径对象
        """
        sub_dir = os.path.join(self.abstract_discourses,sub)
        sub_path = Path(sub_dir)
        sub_files = []
        for file in sub_path.iterdir():
            if file.is_file():
                sub_files.append(file)
        
        print(f"{sub}共有{len(sub_files)}个文件")
        return sub_files
    
    # 读取一个label文中的标签的数量
    def label_num(self,file_path):
        try:
            with open(file_path,"r") as f:
                data = json.load(f)
        except:
            print(f"读取文件{file_path}失败")
            return 0
        return len(data)    
        
    
    # 统计一个子集中所有文件的频数
    def freq_sub(self,sub):
        # 如果已经统计过，直接读取
        freq_file = f"./datasets/{self.dataset_name}_{sub}_freq.json"
        if os.path.exists(freq_file):
            with open(freq_file,"r") as f:
                freq_dict = json.load(f)
            return freq_dict
        
        sub_files = self.get_sub(sub)
        freq_dict = {}
        for file in tqdm(sub_files,desc=f"{sub}数据统计"):
           num = self.label_num(file)
           if num in freq_dict:
               freq_dict[num] += 1
           else:
                freq_dict[num] = 1
        print(f"{sub}数据统计完成")
        print(freq_dict)
        # 保存到文件
        with open(f"./datasets/{self.dataset_name}_{sub}_freq.json","w") as f:
            json.dump(freq_dict,f)
        return freq_dict
    
    
    def plot_freq(self,freq_dict):
        """
        绘制频数图
        """
        num= 15
        plot_dict={}
        for key in freq_dict:
            if 0<int(key)<=num:
                plot_dict[key]=freq_dict[key]
        x = list(range(1,num+1))
        y = list([plot_dict[str(key)] for key in range(1,num+1)])
        plt.bar(x,y)
        # 设置x轴的刻度标签
        plt.xticks(x)  # rotation参数可以旋转标签，便于阅读
        plt.xlabel("话语结构数量")
        plt.ylabel("频数")
        plt.title(f"{self.dataset_name}数据集话语结构数量频数")
        plt.savefig(f"./datasets/{self.dataset_name}_freq.png")
        plt.show()
        
    def analyze_abs_dis(self):
        """
        分析摘要话语结构
        """
        freq_dict=self.freq_sub("train")
        self.plot_freq(freq_dict)

if __name__ == '__main__':
    dataset_name = "pubmed"
    print(f"开始{dataset_name}统计数据集")
    dataset = Dataset(dataset_name)
    dataset.analyze_abs_dis()
    