# 240706
# 根据文本的词袋模型对文本进行建模，然后利用一个神经网络对文本进行分类
# 数据来源京东商城评论，本任务本质是利用神经网络做分类，目标变量的取值是0到1之间的数，并且和等于1，取概率大的作为输出，神经网络需要学习从特征变量到目标变量的映射
# 对于这类问题，输出神经元要和分类结果对应，分为几类就几个输出神经元
# 显然sigmoid函数无法保证输出和为1，这里使用常用的softmax函数，归一化指数函数
# yi=(e^xi)/Σ(e^xi)
# 分类问题的损失函数怎么设计呢？ ————交叉熵，Loss=-ΣlogYi,交叉熵衡量的是网络输出的Yi（概率分布）与one-hot（真实情况，只能属于一类）之间的差别
# 分类问题的特点：1输出单元数等于类别数；2输出单元的值都是（0，1）的数，和等于1；3最后一层是softmax函数；4输出值的最大的类别，或者是根据概率大小输出的类别；5采用交叉熵作为损失函数
# 对于语言模型来说，每个输入的句子的长度不同，这就导致确定输入层神经元数量困难，解决方法：采用词袋模型，词袋是一种文本向量化的方法
# 简单来讲，词袋模型就是将一句话中的所有单词都放进一个袋子（单词表）里，而忽略语法、语义，甚至单词之间的顺序等信息。
# 将所以的词汇添加到词袋中，对句子中出现的词语统计出现次数，并且可以进行归一化，用每一个词出现次数除以句子中总的词数
# 同级目录下的data文件夹包括好评和差评两个数据，这些数据都是句子，需要对语料进行预处理，构建出单词表
# 根据统计，单词表共7133个单词，这就表明输入数据是一个长度为7133的向量，因此输入层是7133个神经元，隐含层还是采用10个，输出层是2个，一个表示好评另一个是差评
# 而词袋模型恰恰就是把握住了这一点，它可以对每一个单词进行计数，这样，只要一个句子中出现了大量正面意义的单词，那么最后的分类就为正面，反之亦然

# 首先需要进行数据预处理，然后进行向量化，接下来划分数据集并训练模型
# 在获得了原始数据后，我们就要对数据进行预处理了，这包括3个步骤：过滤标点符号、分词和建立单词表。
from torch import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import jieba #结巴分词包
import re #正则表达式的包
from collections import Counter #搜集器，可以让统计词频更简单

#数据文件
good_file='data/good.txt'
bad_file='data/bad.txt'
# 过滤标点符号的函数
def filter_punc(sentence):
    sentence=re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return(sentence)


# 接下来扫描文本，建立词袋分辨好评和差评词汇
def Prepare_data(good_file, bad_file, is_filter=True):
    all_words = []  # collect alll the words
    pos_sentences = []  # positive
    neg_sentences = []  # negative
    with open(good_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if is_filter:
                line = filter_punc(line)  # 需要进行过滤标点
            words = jieba.lcut(line)  # 分词
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print('{0} 包含 {1} 行，{2} 个单词.'.format(good_file, idx + 1, len(all_words)))
    count = len(all_words)
    with open(bad_file, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
                words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0} 包含 {1} 行，{2} 个单词.'.format(bad_file, idx + 1, len(all_words) - count))
    # 通过上述操作读取出了好词和坏词并建立的对应的词库
    diction = {}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print('字典大小：{}'.format(len(diction)))
    return (pos_sentences, neg_sentences, diction)
# 调用Prepare_data，完成数据处理工作
pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)
st = sorted([(v[1], w) for w, v in diction.items()])

# print pos_sentences,neg_sentences,diction
# diction内涵的是{'词'：[idx,'出现次数']}

# 查找单词编码idx
def word2index(word,diction):
    if word in diction:
        idx=diction[word][0]
    else:
        idx=-1  #未找到
    return idx

#  编码找单词
def index2word(index,diction):
    for w,v in diction.item():
        if v[0]==index:
            return(w)
        else:
            return None
# 文本数据向量化，定义函数sentence2vec
def sentence2vec(sentence,dictionary):
    vector=np.zeros(len(dictionary))
    for w in sentence:
        vector[w]+=1
    return(1.0 * vector / len(sentence))
# 遍历所有句子，将每一个单词映射成编码
dataset = [] # 数据集
labels = [] # 标签
sentences = [] # 原始句子，用于调试
# 处理正面评论
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0) #正标签为0
    sentences.append(sentence)

# 处理负面评论
for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(1) # 负标签为1
    sentences.append(sentence)

# 打乱所有数据的顺序，形成数据集
# indices为所有数据下标的排列
indices = np.random.permutation(len(dataset))

# 根据打乱的下标，重新生成数据集dataset、标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]
# 将数据集划分为三个，训练集train、验证集val、测试集test，通常情况下是10：1：1
# 验证集的存在是为了检验是否出现了过拟合的情况，需要减少模型的超参数或者提高数据量，从而提升模型的泛化能力
# 首先，在训练模型的时候，是不使用校验集的。
# 其次，在一组超参数下，当我们训练好模型之后，可以利用校验集的数据来测试模型的表现，如果误差与训练数据同样低或差不多，就说明模型的泛化能力很强，否则就说明出现了过拟合的现象
# 可以通过改变超参数、增加dropout层（后面会介绍）或者增加训练数据等方式来避免过拟合
# 训练集用于训练参数，校验集用于调整网络的超参数，比如网络结构、学习率等，测试集用于测试模型的能力。
# 将整个数据集划分为训练集、校验集和测试集，其中校验集和测试集的大小都是整个数据集的十分之一
test_size = int(len(dataset)//10)
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]
#接下来构建神经网络,7133——>10——>2
# 一个简单的前馈神经网络，共3层
# 第一层为线性层，加一个非线性ReLU激活函数，第二层为线性层，中间有10个隐含神经元
# 输入维度为词典的大小：每一条评论的词袋模型
model = nn.Sequential(
    nn.Linear(len(diction), 10),
    nn.ReLU(),
    nn.Linear(10, 2),
    nn.LogSoftmax(dim=1),
)
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素


# ReLU这个函数在输入大于0的时候完全是一个恒等函数，相当于没有完成任何计算，因此，与传统的sigmoid函数相比，它具有计算快、方便反向误差传播等优良特征。
# 同时，由于它在0的位置分成了两个不连续的部分，因此它具备与sigmoid函数同样的非线性特征。该函数特别适合用在深度的前馈神经网络中，计算效果比sigmoid函数好得多。
# 损失函数为交叉熵
cost = torch.nn.NLLLoss()
# 优化算法为Adam，可以自动调节学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
records = []

# 循环10个Epoch
losses = []
for epoch in range(0, 10):
    for i, data in enumerate(zip(train_data, train_label)):
        x, y = data

        # 需要将输入的数据进行适当的变形，主要是要多出一个batch_size的维度，也即第一个为1的维度
        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
        # x的尺寸：batch_size=1, len_dictionary
        # 标签也要加一层外衣以变成1*1的张量
        y = torch.tensor(np.array([y]), dtype=torch.long)
        # y的尺寸：batch_size=1, 1

        # 清空梯度
        optimizer.zero_grad()
        # 模型预测
        predict = model(x)
        # 计算损失函数
        loss = cost(predict, y)
        # 将损失函数数值加入到列表中
        losses.append(loss.data.numpy())
        # 开始进行梯度反传
        loss.backward()
        # 开始对参数进行一步优化
        optimizer.step()

        # 每隔3000步，跑一下校验数据集的数据，输出临时结果
        if i % 3000 == 0:
            val_losses = []
            rights = []
            # 在所有校验数据集上实验
            for j, val in enumerate(zip(valid_data, valid_label)):
                x, y = val
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                predict = model(x)
                # 调用rightness函数计算准确度
                right = rightness(predict, y)
                rights.append(right)
                loss = cost(predict, y)
                val_losses.append(loss.data.numpy())

            # 将校验集合上面的平均准确度计算出来
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                                       np.mean(val_losses),
                                                                                       right_ratio))
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])

print('End of training')
# 绘制误差曲线
import matplotlib.pyplot as plt
a=[i[0] for i in records]
b=[i[1] for i in records]
c=[i[2] for i in records]
plt.figure(figsize=(10,8))
plt.title('the loss or accuracy within training')
plt.plot(a,label='Train loss')
plt.plot(b,label='valid loss')
plt.plot(c,label='valid accuracy')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('loss or accuracy')
plt.show()
# 保存、提取模型（为展示用）
torch.save(model,'./data/bow.mdl')
model = torch.load('./data/bow.mdl')