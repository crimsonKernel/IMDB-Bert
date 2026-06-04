# IMDB-Bert

一个基于微调 BERT 模型实现 IMDB 影评情感分类的个人实践项目

项目围绕 IMDB 评论二分类任务展开，覆盖数据预处理、文本编码、模型微调、效果评估、模型保存与推理调用的完整流程，可作为预训练语言模型在下游文本分类任务中的实践示例。

## 项目简介

- 这是我基于微调 bert 预训练模型实现 IMDB 影评情感分类的一个实践项目，使用 Hugging Face Transformers 完成文本分类训练，并支持本地命令行预测。
- 项目围绕 IMDB 评论二分类任务展开，覆盖数据预处理、文本编码、模型微调、效果评估、模型保存与推理调用的完整流程，体现预
  训练语言模型在下游 NLP 分类任务中的应用。
- 想当初也是参考了好多好多的技术教程才完成，也是感慨啊，转眼过去了那么久，时代也早已经不同了。而补上这份readme竟已是两年之后了，这个项目当时还写了个qt界面的，但是写的简陋，就不放到这里来了。本人写的代码也欢迎大家观看学习和批评指正（那时确实也潦草），哈哈哈哈，还是欢迎来交流的。

## 技术栈

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

## 项目结构

```text
IMDB-Bert/
├─ train.py                 # 模型训练脚本
├─ predict.py               # 单条文本预测
├─ demo.py                  # 命令行预测入口
├─ requirements.txt         # 项目依赖
├─ label_mapping.json       # 标签映射文件
├─ labeledTrainData.tsv     # IMDB 训练数据
├─ IMDB.csv                 # 导出数据文件
├─ bert-base-uncased/       # 本地预训练模型
└─ saved_model/             # 训练后保存的模型
```

## 项目说明

情感分析是自然语言处理中的经典文本分类任务，目标是判断一段文本的情绪倾向。在本项目中，输入数据为 IMDB 电影评论，输出标签为二分类结果：

- 1 -> 正面
- 0 -> 负面

项目使用 `bert-base-uncased` 作为基础预训练模型，并在影评数据上进行微调。相比传统的词袋模型、TF-IDF 或静态词向量方法，BERT 能更充分地利用上下文信息，在处理长文本、语义转折和上下文依赖时具备更强的表示能力。

## 技术原理

### 1. BERT 在文本分类中的作用

BERT 是基于 Transformer Encoder 的双向预训练语言模型，能够从上下文中学习更丰富的语义表示。在文本分类任务中，`BertForSequenceClassification` 会在 BERT 主体之上增加分类层，将句子表示映射为类别 logits，再输出最终分类结果。

本项目的整体流程可以概括为：

1. 输入影评文本
2. 使用 tokenizer 将文本转换为模型输入
3. 送入 BERT 编码器提取上下文特征
4. 通过分类层输出正面或负面的预测结果

### 2. 文本编码与 tokenization

模型不能直接处理原始字符串，因此需要先将文本转换成张量表示。项目中通过 `BertTokenizer` 完成以下步骤：

- 文本分词
- token 到词表 id 的映射
- 截断到固定长度
- padding 补齐
- 生成 `attention_mask`

处理完成后，模型接收的核心输入包括 `input_ids` 和 `attention_mask`。这一步决定了自然语言如何被表示为神经网络可以处理的结构化输入。

### 3. 微调与训练流程

`bert-base-uncased` 本身是通用语言模型，并不直接具备影评情感分类能力，因此需要使用带标签的 IMDB 数据进行微调。训练阶段的目标是让模型学习不同评论在表达方式、上下文组合和情绪倾向上的差异。

项目使用 Hugging Face `Trainer` 搭建训练流程，主要包括：

- 读取本地 IMDB 数据集
- 构造标签映射
- 将数据转换为 `Dataset`
- 对文本进行批量编码
- 划分训练集与验证集
- 执行训练、验证与评估
- 保存模型与 tokenizer

### 4. 分类评估

为了判断模型效果，项目使用了多种常见分类指标：

- `Accuracy`：整体预测正确率
- `Precision`：预测为某类的样本中，有多少是真正属于该类
- `Recall`：真实属于某类的样本中，有多少被成功识别
- `F1-score`：综合衡量 Precision 和 Recall
- `Confusion Matrix`：观察不同类别之间的误判分布

这些指标结合起来，可以更完整地反映模型在二分类任务中的表现，而不是只看单一准确率。

## 使用方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用：

- Python 3.10

### 2. 训练模型

```bash
python train.py
```

训练完成后，模型会保存到 `saved_model/` 目录。

### 3. 命令行预测

```bash
python demo.py
```

也可以直接在代码中调用：

```python
from predict import predict

text = "This movie was amazing and I really liked it."
print(predict(text))
```