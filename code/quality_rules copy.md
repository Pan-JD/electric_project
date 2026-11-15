# 规则参数说明

## --details 在调什么参数时使用
| 筛查项目 | 可调整参数举例 |
|---|---|
| 模糊 | Laplacian 阈值 60、Tenengrad 阈值 900 |
| 曝光 | 过暗均值阈值 50、近黑/近白像素比例 |
| 小目标 | 训练缩放后短边 16 像素、面积占比 0.1%；同时结合 `--imgsz` |
| 低分辨率 | 原图短边 320 像素 |
| 水印覆盖 | OCR 置信度、文字覆盖目标比例 5% |
| 近重复 | 哈希距离、SSIM、颜色差阈值 |
| 大标注框 | 框占图像面积 80% |

例如某张清楚的图片触发模糊，开启 --details 后，可以查看它的清晰度指标及目标位置，再决定如何调整阈值。
它只增加 诊断明细.csv，检测项目和统计结果照常计算。你目前首次全量运行可以省略。调整筛查配置后使用新的输出目录。
## 199 张中的标注相关检查是什么
### 目前实现
| 问题 | 规则能完成什么 |
|---|---|
| 低分辨率、目标过小、精确重复、框坐标异常 | 根据尺寸、哈希和坐标直接统计 |
| 过暗、过曝、模糊 | 根据亮度和清晰度指标筛选候选，结合图片核对阈值 |
| 文字遮挡 | 根据文字位置与目标框的交叠筛选候选，复核是否影响关键内容 |
| 类别错标、漏标、框是否贴合物体 | 需要领域模型预测或人工核对 |

文字遮挡目前的规则是：
1. 用 RapidOCR 预训练模型识别文字和位置，保留置信度 ≥0.6 的结果。
2. 查找水印线索：日期时间、网址、配置关键词、至少 5 张不同图片中相同位置的重复文字，或者跨越目标边界的大字。
3. 计算这些文字区域与已有目标框的交叠，覆盖目标框面积达到 5% 就列入“水印遮挡候选”。
例如，时间水印位于图片角落、没有覆盖目标，就不会触发这项；大字横穿表计读数区域并超过阈值，就会触发。
这里计算的是 OCR 文字区域覆盖目标框的比例，包含文字之间的空白。设备铭牌、表盘刻度也可能与目标交叠，因此目前保留“候选”表述。规则适合缩小复核范围；“是否遮住关键读数”仍需结合具体目标查看。


### 这几项默认开启，使用坐标、文件内容和图像元数据规则：
| 检查项 | 本次数量 | 规则 |
|---|---:|---|
| 标注框几何异常 | 16 | 坐标缺失或非有限值、左右/上下颠倒、零面积；坐标系明确时检查越界 |
| 同图标注版本差异 | 14 | 相同图片内容对应多个标注版本 |
| 图像/标注坐标系待核对 | 49 | EXIF 含旋转/镜像方向、方向值无效，或 XML 声明尺寸与图片尺寸不一致 |
| 标注范围过大候选 | 5 | 框面积达到整图的 80% |

本次历史回放的“标注版本差异”按保存的 XML 文件哈希比较；后续全量 scan 按类别和框坐标比较，忽略目标排列顺序。
这些检查属于标注结构、几何和文件一致性检查，直接读取图片和 label 就能运行。你提到的“把表计标成人员”“目标漏标”“框是否紧贴真实物体”等，需要理解图像内容，后续需领域检测模型和人工复核；当前语义错标/漏标显示“未评估”。
因此，199 张的准确表述是：682 张抽样图像中，199 张触发至少一项筛查规则，占 29.18%，其中包含待复核项。 图像质量与重复问题单独合并为 122 张，占 17.89%。

## 参数说明
| 参数 | 含义 | 对你的任务有什么作用 |
|---|---|---|
| `--layout classified` | 输入是按标签归档的数据；合并跨类别的同图同标注副本 | 各类分别统计，总体避免重复计算归档副本 |
| `--annotation voc` | 按 Pascal VOC 格式读取 XML 标注；默认匹配图片旁的同名 `.xml` | 获取类别、目标坐标、标注图像尺寸 |
| `--imgsz 640` | 按训练输入 **640×640** 的等比例缩放计算目标有效尺寸 | 判断目标缩放后是否过小；原图文件保持原尺寸 |
| `--workers 2` | 使用 2 个并行工作线程读取图片、计算指标和执行 OCR | 控制处理速度与内存占用，仍会扫描全部输入 |
| `--ocr` | 开启 RapidOCR 文字识别 | 支持文字水印与目标交叠检测，使用现成模型，无需自行训练 |
| `--preview-limit 600` | HTML 最多展示 600 张图片，包括问题图和部分对照图 | 方便人工核对；全部统计结果仍写入 CSV |
| `--sample-ratio 0.01` | 按一级类别目录抽样，数量向上取整，每类至少一张，0.1：每类抽取 10%；0.01：每类抽取 1%；默认 1：全量 |
| `--sample-seed 42` |  固定抽样结果；改变种子可换一批。 |

其中 --imgsz 640 用于模拟训练缩放，--ocr 调用现成文字模型；YOLO 的训练需要后续单独开展。
## 运行指令记录
你最终在项目根目录运行下面这条命令即可。它扫描全部分类数据、开启 OCR、生成三张统计清单和最多 60 张图片预览：
``` bash
# 20261104
python -B electronic_data/code/audit_dataset_quality.py scan --source "G:/sample_classified/by_label" --output "electronic_data/run_results/quality_full_640_20261103" --layout classified --annotation voc --imgsz 640 --workers 2 --ocr --preview-limit 600

# 20261105
python -B electronic_data/code/audit_dataset_quality.py scan --source "G:/sample_classified/by_label" --output "electronic_data/run_results/quality_sample_01_ocr" --layout classified --annotation voc --imgsz 640 --workers 6 --sample-ratio 0.1 --sample-seed 42 --ocr --preview-limit 600


```
首次使用空输出目录。--layout classified 保证每类分别统计、总体合并跨类别归档副本；原图和 label 保留原位。

## 前端展示界面
``` bash
# 开启后端，展示结果由--results里的结果文件索引
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104"

# 开启后端，并且使用 --scores-dir 指定大模型标签文件推理文件
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104" --scores-dir "run_results/label_detection" --score-class jyz_pl

# 前后两版评分之间的差异
python -B code/review_dataset_quality.py --results "run_results/quality_full_640_noocr_20261104" --scores-before "run_results/label_detection/旧版评分.csv" --scores-after "run_results/label_detection/新版评分.csv" --score-class jyz_pl
```