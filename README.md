# 指令整理与记录

## 本地：可视化

### 运行指令记录
你最终在项目根目录运行下面这条命令即可，扫描全部分类数据、开启 OCR、生成统计清单
``` bash
# 20261104
python -B electronic_data/code/audit_dataset_quality.py scan --source "G:/sample_classified/by_label" --output "electronic_data/run_results/quality_full_640_20261103" --layout classified --annotation voc --imgsz 640 --workers 2 --ocr --preview-limit 600
```

### 前端展示界面
``` bash
# 开启后端，展示结果由--results里的结果文件索引
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104"

# 开启后端，并且使用 --scores-dir 指定大模型标签文件推理文件
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104" --scores-dir "run_results/label_detection" --score-class jyz_pl

# 前后两版评分之间的差异
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104" --scores-before "run_results/label_detection/旧版评分.csv" --scores-after "run_results/label_detection/新版评分.csv" --score-class jyz_pl

# 评分展示：
# 把服务器的 逐图评分.csv 下载到本地：run_results/sample_filter_v2/逐图评分.csv
# 原来的 --scores-dir 或 --scores-before/--scores-after 参数仍可同时使用
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104" --quality-scores "run_results/sample_filter_v2"

```

## S5 服务器进行训练推理

### 原始数据训练，运行配置与默认输出等参数在对应的json中修改
```bash
# 使用数据集：sample_original_named_seed42，默认输出：outputs/train_original_seed42_gpu2-3_batch256_20261111
bash code/yolo/train_background.sh configs/train_original.json
```

### 硬筛数据训练：
```bash
# 使用数据集：sample_filter_v1_hard_named_seed42，默认输出：outputs/train_filtered_seed42_gpu1_batch256_20261111
bash code/yolo/train_background.sh configs/train.json
```

### 停止对应的训练
```bash
bash code/yolo/train_background.sh configs/train_original.json --stop
bash code/yolo/train_background.sh configs/train.json --stop
```

### 重新查看对应训练日志：
```bash
bash code/yolo/train_background.sh configs/train_original.json --logs
bash code/yolo/train_background.sh configs/train.json --logs
```

- 训练完成后主要保留：
outputs/实验名称/
├── weights/
│   ├── best.pt                # 验证集表现最好的模型
│   └── last.pt                # 最后一轮模型
├── 最终指标.json
├── results.csv               # 各轮损失和指标
├── results.png               # 训练曲线
├── 实验配置.json
├── 数据配置.yaml
├── 环境记录.json
└── evaluation/
    ├── 各类指标.csv           # 各类别 Precision、Recall、F1、mAP
    ├── BoxPR_curve.png
    ├── BoxF1_curve.png
    ├── …其他指标曲线和混淆矩阵
    └── val_batch*.jpg         # 部分样本标注与预测展示

logs/实验名称_时间.log          # 后台运行日志
logs/实验名称_时间.pid          # 后台进程记录
重新做实验前，修改配置中的 output，使用新的实验名称。已有输出目录会阻止重复启动。

### S5：单独评估已有模型
```bash
# 原始模型 last.pt：
python -u -B code/yolo/evaluate.py \
  --model outputs/train_original_seed42_gpu2-3_batch256_20261111/weights/last.pt \
  --data /mnt/data/panjinduo/datasets/electronic_data/sample_original_named_seed42/data_val_no_train_duplicates.yaml \
  --split val \
  --imgsz 640 --batch 256 --device 1 --workers 6 \
  --output outputs/eval_original_last_on_original_val_no_train_duplicates
```

```bash
# 硬筛模型 last.pt：
python -u -B code/yolo/evaluate.py \
  --model outputs/train_filtered_seed42_gpu1_batch256_20261111/weights/last.pt \
  --data /mnt/data/panjinduo/datasets/electronic_data/sample_original_named_seed42/data_val_no_train_duplicates.yaml \
  --split val \
  --imgsz 640 --batch 256 --device 1 --workers 6 \
  --output outputs/eval_filtered_best_on_original_val_no_train_duplicates
```

- 评估结果目录：
outputs/评估名称/
├── 最终指标.json
├── 各类指标.csv
├── 数据配置.yaml
├── BoxPR_curve.png
├── BoxF1_curve.png
├── …其他指标曲线和混淆矩阵
└── val_batch*.jpg

## S5 图像质量评分
### 评分运行指令，启动质量评分流程
```bash
python -u -B code/score_dataset.py --config configs/score.json

- configs/score.json 中：
- annotation_scores_csv: null：完全不使用大模型标注评分。
- weights、scoring：评分权重与规则。
- output：评分结果目录。
```
- 输入数据集为/mnt/data/panjinduo/datasets/electronic_data/sample_filter_v1_hard_named_seed42
- 输入为硬筛数据中的四类样本，评分结果保存在：
electronic_data/quality_scores_v2/
├── 逐图评分.csv
├── 类别评分统计.csv
├── 分数区间统计.csv
├── 评分配置.json
├── 评分汇总.json
└── _cache/experiment.sqlite
- 评分规则、权重和统计区间均在 configs/score.json 中调整

### 评分后划分新的训练数据与测试数据
```bash
python -u -B code/filter_scored_dataset.py --config configs/filter_scored.json

- 筛选阈值在 configs/filter_scored.json：
- "minimum_scores": "图像质量分": 3.5
- 划分策略是继承原数据的 seed=42 训练/验证归属，然后分别筛选两个集合。不会重新随机划分，也不会强制将筛选后的比例再次调整为8∶2。
```

### 评分筛选后的训练流程
```bash
# 评分筛选后的训练
bash code/yolo/train_background.sh configs/train_scored.json
- 每轮验证：关闭，训练结束后统一评估
- 修改运行GPU、batchsize与输出目录

# 查看日志：
bash code/yolo/train_background.sh configs/train_scored.json --logs

# 停止该组训练：
bash code/yolo/train_background.sh configs/train_scored.json --stop

4．单独进行带标注的模型评估

# 下面使用最终一轮的 last.pt，可自行替换为 best.pt：
python -u -B code/yolo/evaluate.py \
  --model outputs/train_scored_filtered_v2_seed42_gpu1_batch256/weights/last.pt \
  --data datasets/scored_filtered_v2/seed_42/data.yaml \
  --split val \
  --imgsz 640 \
  --batch 256 \
  --device 1 \
  --workers 3 \
  --output outputs/eval_scored_image_last_on_val
```

### 加入大模型评分筛选后的训练流程
```bash
# 将现有 fhz_f 和 jyz_pl 评分匹配到硬筛样本，并保留原train/val归属
python -u -B code/split_annotation_scores.py \
  --scores \
    run_results/label_detection/fhz_f_train_scores.csv \
    run_results/label_detection/fhz_f_val_scores.csv \
    run_results/label_detection/jyz_pl_v3_7601_scores.csv \
  --index electronic_data/quality_scores_v2/逐图评分.csv \
  --output run_results/label_detection/current

# 等四类标注评分完整后，再执行：
python -u -B code/score_dataset.py --config configs/score_labels.json

# 按照图像、标注后的综合得分筛选数据集划分
python -u -B code/filter_scored_dataset.py --config configs/filter_scored_labels.json

# 开始带所有样本筛选的模型训练
bash code/yolo/train_background.sh configs/train_scored_labels.json

```

### 本地查看纯图像评分，继续复用现有网页：
```bash
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104" --quality-scores "electronic_data/quality_scores_v2"
```












