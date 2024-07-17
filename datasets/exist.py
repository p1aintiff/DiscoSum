from pathlib import Path
import json
import os

class Datasets:
    def __init__(self, dataset):
        self.set_dir = dataset
        self.names=['abstract-discourses/','content-discourses/','human-abstracts/','inputs/','labels/','section-labels/']


    def load_list(self,sub):
        """
        获取指定sub下的各数据文件名列表
        """
        # inputs
        path = Path(self.set_dir+"inputs/"+sub)
        print("one input", path)
        self.inputs = [file.name for file in path.glob('*.json')]

        # labels
        path = Path(self.set_dir+"labels/"+sub)
        self.labels = [file.name for file in path.glob('*.json')]

        # section-labels
        path = Path(self.set_dir+"section-labels/"+sub)
        self.section_labels = [file.name for file in path.glob('*.json')]

        # abstract-discourses
        path = Path(self.set_dir+"abstract-discourses/"+sub)
        self.abstract_discourses = [file.name for file in path.glob('*.json')]

        # content-discourses
        path = Path(self.set_dir+"content-discourses/"+sub)
        self.content_discourses = [file.name for file in path.glob('*.json')]

        # human-abstracts
        path = Path(self.set_dir+"human-abstracts/"+sub)
        self.human_abstracts = [file.name for file in path.glob('*.txt')]

    def check_one(self,tru):
        f,sub = tru
        # 检查一个input
        with open(self.set_dir+"inputs/"+sub+f,'r') as of:
            try:
                d = json.load(of)
            except:
                print(f"load json {f}")
            try:
                sentences = sum(d['section_lengths'])
            except:
                print(f"no find d? {f}")
                return
            pa = self.set_dir+"labels/"+sub+'/'+f
            # print(pa)
            # 检查label文件是否存在
            if os.path.exists(pa):
                # 检查label文件的句子数量是否匹配
                with open(pa, 'r') as of2:
                    d2 = json.load(of2)
                    labels = d2['labels']
                    if len(labels) == sentences:
                        # 检查其它label文件是否存在
                        if f in self.abstract_discourses and f in self.content_discourses and f.replace(".json",".txt") in self.human_abstracts and f in self.section_labels:
                            # return f.name
                            # 检查content-discourses的数量匹配
                            with open(self.set_dir+"content-discourses/"+sub+'/'+f,'r') as con_dis:
                                con = json.load(con_dis)
                                if len(con)==sentences:
                                    return f
                                else:
                                    print(f"content-discourses {f} not match")
                        else:
                            print(f"lack {f}")
                    else:
                        print(f"labels {f} not match")
            else:
                print(f"no label {f}")
    
    
    def run(self):
        for sub in ["val/","test/", "train/"]:
            print(f"start {sub}")
            self.load_list(sub)
            print(f"inputs 的数量：{len(self.inputs)}")
            for f in self.inputs:
                self.check_one((f,sub))
            print(f"{sub} done")

if __name__ == "__main__":
    # d = Datasets('./arxiv/')
    d = Datasets('./pubmed/')
    d.run()
    # d.load_list("val/")
    # d.check_one("val/")