import pandas as pd
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from datasets import Dataset
import numpy as np

# 读取本地CSV文件
df = pd.read_csv('./labeledTrainData.tsv',sep='\t')

# 选择只保留几千条数据
df = df.sample(n=6000, random_state=92)

# 标签映射
category_mapping = {
    0: "负面",
    1: "正面"
}
df['large_category'] = df['sentiment'].map(category_mapping)

# 查看映射结果
print(df['large_category'].value_counts())

# 将大类别映射为整数标签
unique_labels = ["负面", "正面"]
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
df['label'] = df['large_category'].map(label_mapping)

# # 保存训练集
# df[["review", "sentiment"]].to_csv("IMDB.csv", index=False)
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

# 将Pandas DataFrame转换为 Huggingface Dataset
dataset = Dataset.from_pandas(df)


# 加载预训练的BERT模型和分词器
local_model_path = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # 设置长度补齐
model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=len(set(df['large_category'])))


# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['review'], truncation=True, padding='max_length', max_length=128)



# 手动计算准确率和F1-score等指标的函数
def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    predictions = predictions.argmax(axis=1)
    accuracy = np.mean(predictions == labels)

    # 手动计算Precision, Recall和F1-score
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for label in np.unique(labels):
        tp = np.sum((predictions == label) & (labels == label))
        fp = np.sum((predictions == label) & (labels != label))
        fn = np.sum((predictions != label) & (labels == label))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    f1 = np.mean(f1_scores)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# 数据预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True).train_test_split(test_size=0.2, seed=92)


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # 添加函数
)


# 计算混淆矩阵
def confusion_matrix(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


trainer.train()

# 评估模型
results = trainer.evaluate()
print(f"准确率: {results['eval_accuracy']}")
print(f"精准率: {results['eval_precision']}")
print(f"召回率: {results['eval_recall']}")
print(f"F1-score: {results['eval_f1']}")

# 预测标签
predictions = trainer.predict(tokenized_dataset["test"]).predictions
predicted_labels = np.argmax(predictions, axis=1)
true_labels = tokenized_dataset["test"]['label']
labels = list(label_mapping.values())
cm = confusion_matrix(true_labels, predicted_labels, labels)
print("混淆矩阵:\n", cm)

output_dir = './saved_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型和分词器成功保存到 '{output_dir}'")
