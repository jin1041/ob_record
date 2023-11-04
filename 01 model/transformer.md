### 1.词嵌入`nn.embedding`
将文本或词语映射到一个向量空间的技术。嵌入的目的是捕捉词语或文本的语义信息，并将其表示为数值向量。
```python
import numpy as np
from scipy.spatial.distance import cosine

# 预训练的GloVe词嵌入模型

word_embeddings = {
    'apple': np.array([0.3, 0.6, -0.1, 0.9]),
    'banana': np.array([-0.2, 0.8, 0.5, 0.2]),
    'orange': np.array([0.7, -0.3, 0.6, 0.1]),
    'carrot': np.array([-0.5, 0.2, 0.4, -0.7]),
}

def get_word_embedding(word):
    return word_embeddings.get(word, np.zeros(4))  # 4维的词嵌入向量

# 计算词语之间的相似性

def compute_similarity(word1, word2):
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    similarity = 1 - cosine(embedding1, embedding2)  # 使用余弦相似度计算相似性
    return similarity

  

# 示例使用
word1 = 'apple'
word2 = 'banana'
similarity_score = compute_similarity(word1, word2)
print(f"The similarity between '{word1}' and '{word2}' is: {similarity_score}")
```

### 2.序列长度和编码长度
假设我们有一个机器翻译任务，源语言是英语，目标语言是法语。

源语言句子（英语）： "I love cats."  
目标语言句子（法语）： "J'adore les chats."

现对这两个句子进行编码。将每个单词转换为对应的词嵌入（word embedding）向量。
假设词嵌入向量维度为4，然后我们将每个单词嵌入为相应的向量表示：
```
"I": [0.1, 0.2, 0.3, 0.4]  
"love": [0.5, 0.6, 0.7, 0.8]  
"cats": [0.9, 1.0, 1.1, 1.2]  
"J'adore": [1.3, 1.4, 1.5, 1.6]  
"les": [1.7, 1.8, 1.9, 2.0]  
"chats": [2.1, 2.2, 2.3, 2.4]
```
假设最大序列长度设置为4（`max_src_seq_len = max_tgt_seq_len = 4`），确定序列编码长度（`max_position_len`）。在这种情况下，我们可以选择将`max_position_len`设置为5。使用一个大小为5x4的位置编码矩阵来对输入序列进行编码。

位置编码矩阵示意图：
```
[0.0, 0.0, 0.0, 0.0]
[1.0, 0.1, 0.1, 0.1]
[2.0, 0.2, 0.2, 0.2]
[3.0, 0.3, 0.3, 0.3]
[4.0, 0.4, 0.4, 0.4]
```

现在我们可以将源语言句子和目标语言句子的词嵌入向量与对应的位置编码向量相加，以得到最终的输入序列表示。
对于源语言句子 "I love cats."，将词嵌入向量和位置编码向量相加得到：
```
"I": [0.1, 0.2, 0.3, 0.4] + [0.0, 0.0, 0.0, 0.0] = [0.1, 0.2, 0.3, 0.4]
"love": [0.5, 0.6, 0.7, 0.8] + [1.0, 0.1, 0.1, 0.1] = [1.5, 0.7, 0.8, 0.9]
"cats": [0.9, 1.0, 1.1, 1.2] + [2.0, 0.2, 0.2, 0.2] = [2.9, 1.2, 1.3, 1.4]
```

同样的步骤也适用于目标语言句子 "J'adore les chats."。

* 序列编码长度（`max_position_len`）不同于实际序列的长度，它是作为一个超参数来设置的，用于**为每个位置提供一个编码向量**。这样，模型可以使用这些位置编码向量来捕捉不同位置的信息，并对输入序列进行编码。
### 3.雅可比矩阵
