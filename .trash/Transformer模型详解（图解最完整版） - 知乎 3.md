Transformer模型详解（图解最完整版） - 知乎

[](javascript:void(0))

[](//www.zhihu.com)

首发于[初识CV](//www.zhihu.com/column/c_1186688096946528256)

写文章

![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c)

![](https://picx.zhimg.com/70/v2-7be8fe269991a236f000168291481c8b_1440w.image?source=172ae18b&biz_tag=Post)

Transformer模型详解（图解最完整版）
=======================

[![](https://picx.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=172ae18b)
](//www.zhihu.com/people/AI_team-WSF)

[初识CV](//www.zhihu.com/people/AI_team-WSF)

[​](https://www.zhihu.com/question/48510028)

西安电子科技大学 电子科学与技术硕士

​关注他

*   ![](https://pica.zhimg.com/50/v2-3b02011be580e81e76ed47176c021de2.jpg?source=9f6531fb)
    
*   ![](https://picx.zhimg.com/50/v2-84be2e7c19aa002414a85ce679802556.jpg?source=9f6531fb)
    
*   ![](https://picx.zhimg.com/50/v2-38f1089392919454af52b2df0e8761dd.jpg?source=9f6531fb)
    

陈东文、挂枝儿、Charlie等人赞同

​

目录

收起

前言

1.Transformer 整体结构

2\. Transformer 的输入

2.1 单词 Embedding

2.2 位置 Embedding

3\. Self-Attention（自注意力机制）

3.1 Self-Attention 结构

3.2 Q, K, V 的计算

3.3 Self-Attention 的输出

3.4 Multi-Head Attention

4\. Encoder 结构

4.1 Add & Norm

4.2 Feed Forward

4.3 组成 Encoder

5\. Decoder 结构

5.1 第一个 Multi-Head Attention

5.2 第二个 Multi-Head Attention

5.3 Softmax 预测输出单词

6\. Transformer 总结

> 建议大家看一下李宏毅老师讲解的Transformer，非常简单易懂（个人觉得史上最强transformer讲解）：[https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV\_el3uVTsOK\_ZK5L0Iv_EQoL1JefRL4&index=60](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DugWDIIOHtPA%26list%3DPLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4%26index%3D60)

前言
--

Transformer由论文《Attention is All You Need》提出，现在是谷歌云TPU推荐的参考模型。论文相关的Tensorflow的代码可以从GitHub获取，其作为Tensor2Tensor包的一部分。哈佛的NLP团队也实现了一个基于PyTorch的版本，并注释该论文。

在本文中，我们将试图把模型简化一点，并逐一介绍里面的核心概念，希望让普通读者也能轻易理解。

Attention is All You Need：[Attention Is All You Need](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)

1.Transformer 整体结构
------------------

首先介绍 Transformer 的整体结构，下图是 Transformer 用于中英文翻译的整体结构：

![](https://pic4.zhimg.com/80/v2-4544255f3f24b7af1e520684ae38403f_720w.webp)

Transformer 的整体结构，左图Encoder和右图Decoder

可以看到 **Transformer 由 Encoder 和 Decoder 两个部分组成**，Encoder 和 Decoder 都包含 6 个 block。Transformer 的工作流程大体如下：

**第一步：** 获取输入句子的每一个单词的表示向量 **X**，**X**由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。

![](https://pic4.zhimg.com/80/v2-7dd39c44b0ae45d31a3ae7f39d3f883f_720w.webp)

Transformer 的输入表示

**第二步：** 将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 **x**) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 **C**，如下图。单词向量矩阵用 Xn×dX_{n\\times d} 表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。

![](https://pic3.zhimg.com/80/v2-45db05405cb96248aff98ee07a565baa_720w.webp)

Transformer Encoder 编码句子信息

**第三步**：将 Encoder 输出的编码信息矩阵 **C**传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 **Mask (掩盖)** 操作遮盖住 i+1 之后的单词。

![](https://pic2.zhimg.com/80/v2-5367bd47a2319397317562c0da77e455_720w.webp)

Transofrmer Decoder 预测

上图 Decoder 接收了 Encoder 的编码矩阵 **C**，然后首先输入一个翻译开始符 "<Begin>"，预测第一个单词 "I"；然后输入翻译开始符 "<Begin>" 和单词 "I"，预测单词 "have"，以此类推。这是 Transformer 使用时候的大致流程，接下来是里面各个部分的细节。

2\. Transformer 的输入
-------------------

Transformer 中单词的输入表示 **x**由**单词 Embedding** 和**位置 Embedding** （Positional Encoding）相加得到。

![](https://pic4.zhimg.com/80/v2-b0a11f97ab22f5d9ebc396bc50fa9c3f_720w.webp)

Transformer 的输入表示

### 2.1 单词 Embedding

单词的 Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

### 2.2 位置 Embedding

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。** 所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

位置 Embedding 用 **PE**表示，**PE** 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='136'></svg>)

其中，pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。使用这种公式计算 PE 有以下的好处：

*   使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
*   可以让模型容易地计算出相对位置，对于固定长度的间距 k，**PE(pos+k)** 可以用 **PE(pos)** 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。

将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入。

3\. Self-Attention（自注意力机制）
--------------------------

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='884'></svg>)

Transformer Encoder 和 Decoder

上图是论文中 Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为 **Multi-Head Attention**，是由多个 **Self-Attention**组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。

因为 **Self-Attention**是 Transformer 的重点，所以我们重点关注 Multi-Head Attention 以及 Self-Attention，首先详细了解一下 Self-Attention 的内部逻辑。

### 3.1 Self-Attention 结构

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='406' height='488'></svg>)

Self-Attention 结构

上图是 Self-Attention 的结构，在计算的时候需要用到矩阵**Q(查询),K(键值),V(值)**。在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而**Q,K,V**正是通过 Self-Attention 的输入进行线性变换得到的。

### 3.2 Q, K, V 的计算

Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。** 

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='875'></svg>)

Q, K, V 的计算

### 3.3 Self-Attention 的输出

得到矩阵 Q, K, V之后就可以计算出 Self-Attention 的输出了，计算的公式如下：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='171'></svg>)

Self-Attention 的输出

公式中计算矩阵**Q**和**K**每一行向量的内积，为了防止内积过大，因此除以 d_{k} 的平方根。**Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以 K^{T} ，1234 表示的是句子中的单词。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='228'></svg>)

Q乘以K的转置的计算

得到QK^{T} 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1.

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='247'></svg>)

对矩阵的每一行进行 Softmax

得到 Softmax 矩阵之后可以和**V**相乘，得到最终的输出**Z**。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='217'></svg>)

Self-Attention 输出

上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 Z_{1} 等于所有单词 i 的值 V_{i} 根据 attention 系数的比例加在一起得到，如下图所示：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='216'></svg>)

Zi 的计算方法

### 3.4 Multi-Head Attention

在上一步，我们已经知道怎么通过 Self-Attention 计算得到输出矩阵 Z，而 Multi-Head Attention 是由多个 Self-Attention 组合形成的，下图是论文中 Multi-Head Attention 的结构图。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='859'></svg>)

Multi-Head Attention

从上图可以看到 Multi-Head Attention 包含多个 Self-Attention 层，首先将输入**X**分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵**Z**。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵**Z**。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='831'></svg>)

多个 Self-Attention

得到 8 个输出矩阵 Z_{1} 到 Z_{8} 之后，Multi-Head Attention 将它们拼接在一起 **(Concat)**，然后传入一个**Linear**层，得到 Multi-Head Attention 最终的输出**Z**。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='388'></svg>)

Multi-Head Attention 的输出

可以看到 Multi-Head Attention 输出的矩阵**Z**与其输入的矩阵**X**的维度是一样的。

4\. Encoder 结构
--------------

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='884'></svg>)

Transformer Encoder block

上图红色部分是 Transformer 的 Encoder block 结构，可以看到是由 Multi-Head Attention, **Add & Norm, Feed Forward, Add & Norm** 组成的。刚刚已经了解了 Multi-Head Attention 的计算过程，现在了解一下 Add & Norm 和 Feed Forward 部分。

### 4.1 Add & Norm

Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='107'></svg>)

Add &amp;amp;amp;amp;amp; Norm 公式

其中 **X**表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(**X**) 和 FeedForward(**X**) 表示输出 (输出与输入 **X** 维度是一样的，所以可以相加)。

**Add**指 **X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='117'></svg>)

残差连接

**Norm**指 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛。

### 4.2 Feed Forward

Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='596' height='77'></svg>)

Feed Forward

**X**是输入，Feed Forward 最终得到的输出矩阵的维度与**X**一致。

### 4.3 组成 Encoder

通过上面描述的 Multi-Head Attention, Feed Forward, Add & Norm 就可以构造出一个 Encoder block，Encoder block 接收输入矩阵 X_{(n\\times d)} ，并输出一个矩阵 O_{(n\\times d)} 。通过多个 Encoder block 叠加就可以组成 Encoder。

第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block 输出的矩阵就是**编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='900'></svg>)

Encoder 编码句子信息

5\. Decoder 结构
--------------

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='884'></svg>)

Transformer Decoder block

上图红色部分为 Transformer 的 Decoder block 结构，与 Encoder block 相似，但是存在一些区别：

*   包含两个 Multi-Head Attention 层。
*   第一个 Multi-Head Attention 层采用了 Masked 操作。
*   第二个 Multi-Head Attention 层的**K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 Decoder block 的输出计算。
*   最后有一个 Softmax 层计算下一个翻译单词的概率。

### 5.1 第一个 Multi-Head Attention

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。

下面的描述中使用了类似 Teacher Forcing 的概念，不熟悉 Teacher Forcing 的童鞋可以参考以下上一篇文章Seq2Seq 模型详解。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='140'></svg>)

Decoder 预测

Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 (<Begin> I have a cat) 和对应输出 (I have a cat <end>) 传递到 Decoder。那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "<Begin> I have a cat <end>"。** 

**第一步：** 是 Decoder 的输入矩阵和 **Mask** 矩阵，输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，**Mask** 是一个 5×5 的矩阵。在 **Mask** 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='277'></svg>)

输入矩阵与 Mask 矩阵

**第二步：** 接下来的操作和之前的 Self-Attention 一样，通过输入矩阵**X**计算得到**Q,K,V**矩阵。然后计算**Q**和 K^{T} 的乘积 QK^{T} 。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='240'></svg>)

Q乘以K的转置

**第三步：** 在得到 QK^{T} 之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用**Mask**矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='206'></svg>)

Softmax 之前 Mask

得到 **Mask** QK^{T} 之后在 **Mask** QK^{T}上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。

**第四步：** 使用 **Mask** QK^{T}与矩阵 **V**相乘，得到输出 **Z**，则单词 1 的输出向量 Z_{1} 是只包含单词 1 信息的。

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='292'></svg>)

Mask 之后的输出

**第五步：** 通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Z_{i} ，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出Z_{i} 然后计算得到第一个 Multi-Head Attention 的输出**Z**，**Z**与输入**X**维度一样。

### 5.2 第二个 Multi-Head Attention

Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的 **K, V**矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 **Encoder 的编码信息矩阵 C** 计算的。

根据 Encoder 的输出 **C**计算得到 **K, V**，根据上一个 Decoder block 的输出 **Z** 计算 **Q** (如果是第一个 Decoder block 则使用输入矩阵 **X** 进行计算)，后续的计算方法与之前描述的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 **Mask**)。

### 5.3 Softmax 预测输出单词

Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如下：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='239'></svg>)

Decoder Softmax 之前的 Z

Softmax 根据输出矩阵的每一行预测下一个单词：

![](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='357'></svg>)

Decoder Softmax 预测

这就是 Decoder block 的定义，与 Encoder 一样，Decoder 是由多个 Decoder block 组合而成。

6\. Transformer 总结
------------------

*   Transformer 与 RNN 不同，可以比较好地并行训练。
*   Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
*   Transformer 的重点是 Self-Attention 结构，其中用到的 **Q, K, V**矩阵通过输出进行线性变换得到。
*   Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。

[\[1\]](#ref_1)[\[2\]](#ref_2)

参考
--

1.  [^](#ref_1_0)论文:Attention Is All You Need [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  [^](#ref_2_0)Transformer 模型详解 [https://baijiahao.baidu.com/s?id=1651219987457222196&wfr=spider&for=pc](https://baijiahao.baidu.com/s?id=1651219987457222196&wfr=spider&for=pc)

编辑于 2021-11-23 11:01

「真诚赞赏，手留余香」

赞赏

5 人已赞赏

[![](https://pica.zhimg.com/v2-99b0fbf7ae2898197a96cdba767d8d78_l.jpg?source=d16d100b)
](//www.zhihu.com/people/chip-71-78)[![](https://pic1.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=d16d100b)
](//www.zhihu.com/people/li-cheng-85-70)[![](https://picx.zhimg.com/c8f182e7f8125367e60f93e0188bf82c_l.png?source=d16d100b)
](//www.zhihu.com/people/liu_jian_0413)[![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=d16d100b)
](//www.zhihu.com/people/whyme-23-93)[![](https://pica.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=d16d100b)
](//www.zhihu.com/people/san-qi-32-88)

[

Transformer

](//www.zhihu.com/topic/20746363)

[

自然语言处理

](//www.zhihu.com/topic/19560026)

[

深度学习（Deep Learning）

](//www.zhihu.com/topic/19813032)

​赞同 8003​​382 条评论

​分享

​喜欢​收藏​申请转载

​

![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c)

发布一条带图评论吧

  

382 条评论

默认

最新

[![](https://picx.zhimg.com/v2-3aaa1359ee01a8196608f3bef68fed40_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/b9bc5b5732ef982a9bdccdbd20109128)

[柚子绿茶tea](https://www.zhihu.com/people/b9bc5b5732ef982a9bdccdbd20109128)

感谢分享，这个transformer是我见过讲解的最明白的了。![](https://pic2.zhimg.com/v2-419a1a3ed02b7cfadc20af558aabc897.png)

2021-09-02

​回复​103

[![](https://picx.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

[初识CV](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

作者​

不客气哈，谢谢！

2021-09-03

​回复​20

[![](https://picx.zhimg.com/71e30c0eb_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/08263dbaf9f47a95be261b7a4dee4dd3)

[ds mi](https://www.zhihu.com/people/08263dbaf9f47a95be261b7a4dee4dd3)

[口舌言口](https://www.zhihu.com/people/2038356c712a557b35c33c6aca88f6f5)

ChatGPT, Chat Generative Pre-training Transformer

05-15

​回复​10

展开其他 3 条回复​

[![](https://pic1.zhimg.com/v2-9a5dbf6f949fe07d2710bd4c0173e0f3_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/b119b89584499aaf633266eb6af6706b)

[个耿耿于怀](https://www.zhihu.com/people/b119b89584499aaf633266eb6af6706b)

写的挺好的

2021-10-10

​回复​31

[![](https://picx.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

[初识CV](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

作者​

![](https://pic1.zhimg.com/v2-0942128ebfe78f000e84339fbb745611.png)

2021-10-11

​回复​5

[![](https://pic1.zhimg.com/v2-c1812254bd5eec8684beee19f13405d3_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/af5ab744f853854fb468ef5457bbdc0a)

[傲来国主](https://www.zhihu.com/people/af5ab744f853854fb468ef5457bbdc0a)

确实很好

2021-10-13

​回复​3

展开其他 1 条回复​

[![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/434defd2930307ea6fa1e5fe6411f965)

[时光似流年](https://www.zhihu.com/people/434defd2930307ea6fa1e5fe6411f965)

你好，对于 transformer的 decoder的有点疑惑，现在公开的代码测试的时候， decoder也需要输入真实标签的 embedding，而并不是decoder已经真实输出的embedding，而实际上我们预测时是不知道待预测数据的真实标签的，为什么大家测试的时候没有写一个不需要输入真实标签的预测部分 decoder代码呢？而是测试和训练共用一个预测通道。这种方式对于没有真实标签的待预测数据是没法进行预测的吧？即使测试集有真实标签，计算测试集指标的时候，测试时输入了测试集数据真实标签的 embedding，这样得出来的结果不是类似于“作弊”的结果么？因为如果待预测数据不知道真实标签，那么其中某个单词预测错误后面可能都会跟着错，而输入真实标签的话，即使上一个单词预测错了，模型也会强制输入待预测单词之前所有正确单词的 embedding，这就导致后面很大可能一些单词被预测正确，就导致测试指标好了，但是这样得出来的结果有点太虚伪了。还是我对模型理解有问题，希望看到的明白的人能指导一下，感激不尽！

2021-08-10

​回复​27

[![](https://picx.zhimg.com/v2-8a7ea4d791b26ca22de6365ebae0cdea_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/d3d22a9ca5969f408faaa187f7ca9751)

[Isaac Zhang](https://www.zhihu.com/people/d3d22a9ca5969f408faaa187f7ca9751)

[HelloMonica](https://www.zhihu.com/people/957282765e655f4501c27390870c4004)

谢谢leon和作者的解答，我最近搞懂了  

  

  
训练时：第i个decoder的输入 = encoder输出 + ground truth embeding  
预测时：第i个decoder的输入 = encoder输出 + 第(i-1)个decoder输出  

  

  
训练时因为知道ground truth embeding，相当于知道正确答案，网络可以一次训练完成。  
预测时，首先输入start，输出预测的第一个单词 然后start和新单词组成新的query，再输入decoder来预测下一个单词，循环往复 直至end

2021-09-20 · 热评

​回复​157

[![](https://picx.zhimg.com/v2-fbd93a457d00374bc5dd9479474f4ecc_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/1830035cff0792baa7164fdade757eb2)

[呵呵](https://www.zhihu.com/people/1830035cff0792baa7164fdade757eb2)

[樱花雨](https://www.zhihu.com/people/d11fb2189ace4cb075833999ea402c4e)

设K是\[len\_encoder,dk\]，V是\[len\_encoder,dv\]，Q是\[len\_decoder,dq\]，首先是Q和K的转置做矩阵乘法计算注意力（需要确保dk和dq一样），生成了注意力矩阵\[len\_decoder,len\_encoder\]，softmax和伸缩后，每一行的和为1，表示解码器每一个词对编码器每一个单词的注意力分配。之后再和V做矩阵乘法，形成\[len\_decoder,dv\]。  
  
回到你的问题，编码器和解码器长度不一样，不影响其计算，需要注意的是dk和dq一样，dv可以和他们不一样，最后的y的维度和dv一致。多头注意力最后还有一个线性层，可以确保最后输出的维度又和输入X一样。

2022-05-12

​回复​16

查看全部 26 条回复​

[![](https://picx.zhimg.com/077fd44ed6c4b81575c8f55177b92f27_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/316c7195fba301e48b29878a367c404c)

[echohandsome](https://www.zhihu.com/people/316c7195fba301e48b29878a367c404c)

Wq,Wk,Wv这个三个矩阵一开始是直接随机初始化得到的吗

2021-07-26

​回复​23

[![](https://picx.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

[初识CV](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

作者​

随机初始化得到的。

2021-07-26

​回复​18

[![](https://pica.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

[初识CV](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

作者​

它在训练过程中就行更新，和其他参数一样根据梯度下降进行更新。

2021-07-26

​回复​11

展开其他 2 条回复​

[![](https://picx.zhimg.com/v2-0e2454a13bb4fe947b711edd319ba4fb_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/b6e4770d937d28662d101f62b35b5416)

[恋金术矢](https://www.zhihu.com/people/b6e4770d937d28662d101f62b35b5416)

请问Decoder部分的Output(Output Embedding)是什么内容？是输入样本X还是Encoder部分的输出C呢

2021-09-30

​回复​8

[![](https://pica.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

[初识CV](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

作者​

根据 Encoder 的输出 C计算得到 K, V，根据上一个 Decoder block 的输出 Z 计算 Q (如果是第一个 Decoder block 则使用输入矩阵 X 进行计算)，

2021-11-05

​回复​8

[![](https://pic1.zhimg.com/v2-a264efdea0ae79be769405d264c6c98c_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/e1c17585475ac344b3867bf2546f1ef6)

[你最珍贵](https://www.zhihu.com/people/e1c17585475ac344b3867bf2546f1ef6)

input embedding和output embedding 是通过token embedding matrix 得到的，需要先构建token embedding matrix，维度是vocab\_size乘d\_model（vocab\_size是词汇量的大小，d\_model是词向量的长度，也就是说每一个词都用一个d\_model维的向量表示，所有的词合起来就构成了这个矩阵），然后用tf.nn.embeddinglookup函数在matrix里按照x查询得到input embedding，x是由词的id构成的向量，设x的长度为T1，那么最终得到的input embedding的维数就是T1乘d\_model。output embedding和input embedding的得到方式相同，不同的地方在于把x换成decoder\_inputs，以机器翻译为例，要把T1长的英文翻译成T2长的中文，那么x是T1长的英文词id，decoder inputs就是对应T2长的中文词id，所以最终得到的output embedding是T2乘d\_model维。

2022-11-23

​回复​4

展开其他 3 条回复​

[![](https://pica.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/685dff3877e7ff470e604157bb4d4a61)

[映摄江山](https://www.zhihu.com/people/685dff3877e7ff470e604157bb4d4a61)

多头注意力机制那儿怎么感觉不对呢。X乘以每一个变换矩阵得到多头注意力的各个输入不对吧。正确的应该是X乘以Wq得到的矩阵，然后以词向量编码维度那儿拆分成多个头的

2022-10-27

​回复​9

[![](https://picx.zhimg.com/v2-10cd068236c6abe88a1b9d1a7787061e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/ec4ca4cf3b282adee49c2720a3eb1b26)

[evan](https://www.zhihu.com/people/ec4ca4cf3b282adee49c2720a3eb1b26)

终于有人发现讲错了

05-28

​回复​1

[![](https://pic1.zhimg.com/v2-8ad3d9f84724cb020a297138fcc89e82_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/1480e06c87d473c485bb112216c310a6)

[情报对账](https://www.zhihu.com/people/1480e06c87d473c485bb112216c310a6)

论文原文描述多头顺序跟这位作者是一致的，transformer-pytorch源码是一次线性变换后再拆成h个头，我觉得只是为了编码方便（不太确定会不会并行加速），其实没有影响。

10-05

​回复​喜欢

展开其他 1 条回复​

[![](https://picx.zhimg.com/v2-9da86106d761ee8faecb08848d884da5_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/4cb57ed5fcebf9a149549d3c6875ac85)

[潇潇风雨](https://www.zhihu.com/people/4cb57ed5fcebf9a149549d3c6875ac85)

讲的真好！不过有两点不太理解，劳烦解惑  

  

  
第一个问题是decoder的输入，看起来输入像是一次性<Begin> I have a cat这五个单词的表示向量，还是先输入<Begin>，预测出I后再输入<Begin> I 这样循环呢？  

  

  
第二个问题是，“Softmax 根据输出矩阵的每一行预测下一个单词”是怎么实现的呢？这里采用了什么方式把输出矩阵的行向量映射到相应的单词呢？  

  

  
求解惑！万分感谢！

2021-07-09

​回复​9

[![](https://picx.zhimg.com/v2-27c4ae0e1b153a763ef65263b4821634_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/8ecf662d9ef2ec23f2fdc3942d4438ef)

[神经蛙](https://www.zhihu.com/people/8ecf662d9ef2ec23f2fdc3942d4438ef)

第一个问题，我的理解是其实输入了一个5行的矩阵，表示5个训练数据，第一个分类是用begin预测I，第2个是用 begin I预测 have，以此类推，这样可以极大的提高训练效率，把串行的预测转化为了并行  
第二个问题其实和word2vec一样吧，每个单词都对应词表中的一个位置，softmax就是最大化那个位置的词吧

2022-02-28

​回复​11

[![](https://picx.zhimg.com/v2-16340cfaf16380019c183d160df3bb5e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

[初识CV](https://www.zhihu.com/people/7d0bf3ef6a7f044754895a3752969515)

作者​

第一个问题：在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。  
第二问题：可以看一下详解的softmax部分，他其实就是讲单词转换成矩阵，行向量代表着单词的类型，输出概率最大的那个位置就是预测的单词。行向量中单词的位置是固定的，只需要找位置信息就能找到相应的单词了。

2021-07-09

​回复​7

查看全部 9 条回复​

[![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/a28233db724ad90a72058aa5cbcb56da)

[111](https://www.zhihu.com/people/a28233db724ad90a72058aa5cbcb56da)

关于Decoder中第二个Multi-Head Attention说，Q是由上一个decoder block的输出z计算Q，我觉得是不是表达有误，应该是同一个decoder block的第一个Masked Multi-Head Attention的输出z计算Q。

2022-10-13

​回复​6

[![](https://picx.zhimg.com/v2-27b12f17d6f70a33706d8d1ac4ce8d71_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/18996f80b570eb52a2f1fe40562d44a0)

[None](https://www.zhihu.com/people/18996f80b570eb52a2f1fe40562d44a0)

[Mself](https://www.zhihu.com/people/aae044c1c3bd0ead6421120ee19a9309)

<begin> 我理解在预测时decoder的输入第一个是<begin>， 通过<begin>预测得到I的向量， 在和<begin>拼接起来输入，模拟训练时的mask操作

09-13

​回复​喜欢

[![](https://picx.zhimg.com/ff09c161c_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/aae044c1c3bd0ead6421120ee19a9309)

[Mself](https://www.zhihu.com/people/aae044c1c3bd0ead6421120ee19a9309)

![](https://pica.zhimg.com/v2-4812630bc27d642f7cafcd6cdeca3d7a.jpg?source=88ceefae)

[江天雪意云缭乱](https://www.zhihu.com/people/a29aa0a3520321b884a9d7eb5929bbca)

这边也有一个疑问，最开始第一个单词的Q是哪里来的？

08-06

​回复​喜欢

展开其他 1 条回复​

[![](https://picx.zhimg.com/v2-91814dd2adc527bb9e51bfb93a8fab0e_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/7a82078714f41becd6b2a4bfb9bc6a8d)

[香鲸资产马司令](https://www.zhihu.com/people/7a82078714f41becd6b2a4bfb9bc6a8d)

知乎炫风格。前面就没看懂，后面更没耐心看懂😭😭

2021-10-17

​回复​4

[![](https://pic1.zhimg.com/v2-791100f411df197ee2dd7bc32aaf1d78_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/6a3a922bb0402381f1c32765e99acb6d)

[你好像可爱了](https://www.zhihu.com/people/6a3a922bb0402381f1c32765e99acb6d)

博主写的很好懂呀 建议看完原文再来看 会茅塞顿开![](https://pic2.zhimg.com/v2-7f09d05d34f03eab99e820014c393070.png)

02-27

​回复​10

[![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/021fbbf8a1ffab41d306a82ef238b17e)

[橘子风](https://www.zhihu.com/people/021fbbf8a1ffab41d306a82ef238b17e)

[Artorias](https://www.zhihu.com/people/9e2a9400d8a8f4b68fa9d89126a3c054)

deep_thoughts

10-25

​回复​1

查看全部 9 条回复​

[![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/ac91ca85a4f1299e8d8039d657097e57)

[pf67](https://www.zhihu.com/people/ac91ca85a4f1299e8d8039d657097e57)

这文章在描述多头注意力机制问题上个人觉得有一些错误，可能会造成误解，具体就是每个多头实际上是要用model/h来拆分维度的，得到的Z也是model/h维度，所以如果按照论文方案不可能出现Multi-Head Attention图示 的8个维度和单头一致的Z输出叠加（实际上图例的编码维度也不能支持分8个多头）

04-12

​回复​4

[![](https://picx.zhimg.com/v2-f361e66259062c79081594afdb389f2d_l.jpg?source=06d4cd63)
](https://www.zhihu.com/people/285a567562d9c2eeca53f4a033bb753a)

[jvvnn](https://www.zhihu.com/people/285a567562d9c2eeca53f4a033bb753a)

工程实现是这么做的

07-20

​回复​喜欢

点击查看全部评论

![](https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c)

发布一条带图评论吧

  

### 文章被以下专栏收录

[

![](https://pica.zhimg.com/4b70deef7_l.jpg?source=172ae18b)


](//www.zhihu.com/column/c_1186688096946528256)

[

初识CV

](//www.zhihu.com/column/c_1186688096946528256)
--------------------------------------------------------

从这里开始认识人类的眼睛——计算机视觉

[

![](https://picx.zhimg.com/v2-674781ef13a310d6045598d915896623_l.jpg?source=172ae18b)


](//www.zhihu.com/column/c_1173652984163610624)

[

南湖研究院

](//www.zhihu.com/column/c_1173652984163610624)
---------------------------------------------------------

数据竞赛经验，工作积累的笔记，南湖边有可爱的橘猫

[

![](https://pic1.zhimg.com/4b70deef7_l.jpg?source=172ae18b)


](//www.zhihu.com/column/c_1339338855846608896)

[

深度视觉与自然语言探究

](//www.zhihu.com/column/c_1339338855846608896)
---------------------------------------------------------------

记录不断发展的计算机知识。

### 推荐阅读

[

![](https://pic1.zhimg.com/v2-b42f7e65458d2dd989d372e8f62e7d32_250x0.jpg?source=172ae18b)

Transformer在3D语义分割中的应用
======================

在座皆佬



](https://zhuanlan.zhihu.com/p/398833485)[

![](https://picx.zhimg.com/v2-1160ca02935c53dd7a3382021c4d89bd_250x0.jpg?source=172ae18b)

Transformer 超详细解读，一图胜千言
=======================

新智元发表于新智元



](https://zhuanlan.zhihu.com/p/214119876)[

![](https://picx.zhimg.com/v2-1160ca02935c53dd7a3382021c4d89bd_250x0.jpg?source=172ae18b)

Transformer 超详细解读，一图胜千言
=======================

华来知识



](https://zhuanlan.zhihu.com/p/205496205)[

transformer 详细图解
================

transformer 详细图解本文建立在http://jalammar.github.io/illustrated-transformer/博文的基础上， 结合自己的理解学习，如有不当之处，还请同学们指正。 可以任意转载，但转载请说明引用…

早睡早起的小码农



](https://zhuanlan.zhihu.com/p/58408041)

_想来知乎工作？请发送邮件到 jobs@zhihu.com_

×

拖拽到此处

图片将完成下载