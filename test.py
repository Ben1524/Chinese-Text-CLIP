import numpy
from transformers import CLIPProcessor
from component.model import TextCLIPModel
from PIL import Image
import requests

model_name_or_path = 'my_chinese_clip'
# 加载预训练模型权重
model = TextCLIPModel.from_pretrained(model_name_or_path)
CLIPProcessor.tokenizer_class = 'BertTokenizerFast'
# 初始化processor
processor = CLIPProcessor.from_pretrained(model_name_or_path)
# 预处理输入
url = "images/test/dog.jpeg"
image = Image.open(url)

texts=["坐着的小狗", "棕色的熊"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
inputs.pop('token_type_ids')    # 输入中不包含token_type_ids

outputs = model(**inputs)

# 对于每张图片，计算其与所有文本的相似度
logits_per_image = outputs.logits_per_image  # image-text的相似度得分
probs = logits_per_image.softmax(dim=1)  # 对分数进行归一化
print(texts, probs)



