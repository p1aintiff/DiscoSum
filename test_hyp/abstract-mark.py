import os
from pathlib import Path
import json
dir = "test_hyp/arxiv-multi_sent_discourse_summ-both-10-64-32-32-1.5-1"
path = Path(dir)
files = list(path.iterdir())

# 
content = {}
for file_name in files:
    with open(file_name, 'r') as file:
        # 分行读取，每一行储存在list中
        lines = file.readlines()
        # 遍历行
        for line in lines:
            flag = True
            # 检查行开始是否为字母，结尾是否为句号"."
            line = line.strip()
            if line[0]=="," or not line.endswith("."):
                line[0].isalpha()
                flag = False
                print(line[0])
                print(line[-1])
                break
        # 如果没有break，说明所有行都符合要求
        # 将文件名，文件内容储存在dict
        if flag:
            content[file_name.name] = lines
        else:
            print("Error: ", file_name.name)

# 保存到新文件
output_dir = "test_hyp/"
with open(output_dir + "abstract-mark.json", 'w') as file:
    file.write(json.dumps(content,indent=4))
    
