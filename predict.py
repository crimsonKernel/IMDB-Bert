from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

# 加载保存的模型和分词器
output_dir = './saved_model'  # 替换为你保存模型的路径
tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)

# 加载类别映射
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

# 反向映射标签
reverse_label_mapping = {int(v): k for k, v in label_mapping.items()}


# 预测函数
def predict(text):
    # 对输入文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 禁用梯度计算
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

    # 确保预测类别在映射中
    if predicted_class_id in reverse_label_mapping:
        predicted_label = reverse_label_mapping[predicted_class_id]
    else:
        predicted_label = "Unknown"

    return predicted_label

