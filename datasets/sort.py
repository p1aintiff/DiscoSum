import json
import csv

# arxiv数据读取
axiv_js = json.load(open('datasets/arxiv_train_freq.json', 'r'))
sorted_keys = [int(k) for k in axiv_js.keys()]
# print(sorted_keys.sort())
# print(sorted(sorted_keys))

sorted_dict = {}

for k in sorted(sorted_keys):
    sorted_dict[str(k)] = axiv_js[str(k)]
    
print(sorted_dict)

# 保存
with open('datasets/arxiv_train_freq_sorted.json', 'w') as f:
    json.dump(sorted_dict, f, indent=4)
# 保存为csv
with open('datasets/arxiv_train_freq_sorted.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'freq'])
    for k, v in sorted_dict.items():
        writer.writerow([k, v])

# pubmed数据读取
pubmed_js = json.load(open('datasets/pubmed_train_freq.json', 'r'))
sorted_keys = [int(k) for k in pubmed_js.keys()]
# print(sorted_keys.sort())
# print(sorted(sorted_keys))

sorted_dict = {}

for k in sorted(sorted_keys):
    sorted_dict[str(k)] = pubmed_js[str(k)]
    
print(sorted_dict)

# 保存
with open('datasets/pubmed_train_freq_sorted.json', 'w') as f:
    json.dump(sorted_dict, f, indent=4)
    
# 保存为csv
with open('datasets/pubmed_train_freq_sorted.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'freq'])
    for k, v in sorted_dict.items():
        writer.writerow([k, v])
        