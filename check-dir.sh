# 检查目录结构是否完整
# 作者: szc
# 时间: 2024-07-17
#!/bin/bash

# 定义目录列表
directories=(
    "pretrained_embeddings"
    "pretrained_models"
    "test_pyrouge"
    "eval_hyp"
    "temp"
    "log/arxiv"
    "log/pubmed"
    "pred_content_plan"
    "results"
    "test_hyp"
    "datasets"
)

# 循环检查并创建目录
for dir in "${directories[@]}"
do
    if [ ! -d "$dir" ] && [ ! -f "$dir" ]; then
        mkdir -p "$dir"
        echo "Directory $dir created."
    else
        echo "Directory $dir already exists."
    fi
done