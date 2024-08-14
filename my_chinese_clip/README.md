
## CLIP-Vit-Bert-Chinese pretrained model
这是中文版本的CLIP预训练模型，基于LiT-tuning（Locked-image Text tuning）的策略，使用140万中文图文对数据进行多模态对比学习预训练。

Github: [CLIP-Chinese](https://github.com/yangjianxin1/CLIP-Chinese)

Bolg: [CLIP-Chinese：中文多模态对比学习CLIP预训练模型](https://mp.weixin.qq.com/s/6gQX91M-Lt7eiMimhYRJEw)

## Model and Training Detail
该模型主要由文本编码器与图像编码器组成，其中文本编码器为Bert，图像编码器为Vit，我们将该模型称为BertCLIP模型。训练时，Bert使用Langboat/mengzi-bert-base的权重进行初始化，Vit使用openai/clip-vit-large-patch32
的权重进行初始化。采用LiT-tuning（Locked-image Text tuning）的策略进行训练，也就是冻结Vit的权重，训练BertCLIP模型剩余的权重。

## Usage
首先将项目clone到本地，并且安装依赖包
```bash
git clone https://github.com/yangjianxin1/CLIP-Chinese
pip install -r requirements.txt
```

使用如下脚本，就可成功加载预训练权重，对图片和文本进行预处理，并且得到模型的输出
```python
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
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=["一只小狗在摇尾巴", "一只小猪在吃饭"], images=image, return_tensors="pt", padding=True)
inputs.pop('token_type_ids')    # 输入中不包含token_type_ids

outputs = model(**inputs)

# 对于每张图片，计算其与所有文本的相似度
logits_per_image = outputs.logits_per_image  # image-text的相似度得分
probs = logits_per_image.softmax(dim=1)  # 对分数进行归一化

# 对于每个文本，计算其与所有图片的相似度
logits_per_text = outputs.logits_per_text  # text-image的相似度得分
probs = logits_per_text.softmax(dim=1)  # 对分数进行归一化

# 获得文本编码
text_embeds = outputs.text_embeds
# 获得图像编码
image_embeds = outputs.image_embeds
```

单独加载图像编码器，进行下游任务
```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPVisionModel

model_name_or_path = 'my_chinese_clip'
model = CLIPVisionModel.from_pretrained(model_name_or_path)
CLIPProcessor.tokenizer_class = 'BertTokenizerFast'
processor = CLIPProcessor.from_pretrained(model_name_or_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output 
```

单独加载文本编码器，进行下游任务

```python
from component.model import BertCLIPTextModel
from transformers import BertTokenizerFast

model_name_or_path = 'my_chinese_clip'
model = BertCLIPTextModel.from_pretrained(model_name_or_path)
tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)

inputs = tokenizer(["一只小狗在摇尾巴", "一只小猪在吃饭"], padding=True, return_tensors="pt")
inputs.pop('token_type_ids')  # 输入中不包含token_type_ids

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output
```