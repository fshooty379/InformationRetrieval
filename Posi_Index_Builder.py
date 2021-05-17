import math
import os
import nltk
from nltk.stem import PorterStemmer
from natsort import natsorted
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


class Posi_Index_Builder:
    def __init__(self, folderName):
        self.folderName = folderName
        nltk.download('punkt')
        nltk.download('stopwords')
        # 引入英文停用词表
        self.stopWords = set(stopwords.words('english'))
        # 词干提取
        self.stemmer = PorterStemmer()
        # 使用正则分词器
        self.tokenizer = RegexpTokenizer(r'\w+')
        # 语料库
        self.corpus = []
        # 分词、去停用词后的语料库
        self.corpusNormalizedAndTokenized = []
        # 词干索引列表
        self.positionalIndex = defaultdict(list)

    # 读取文件内容
    def readFile(self, fileName):
        notebook_dirctory = os.getcwd()
        file = open(notebook_dirctory + "/" + self.folderName + "/" + fileName, 'r', encoding='utf-8')
        dataFromFile = file.read()
        file.close()
        self.corpus.append(dataFromFile)

    # 进行正则分词、去停用词
    def tokenizeAndNormalize(self):
        # 进行正则分词
        for i in range(0, len(self.corpus)):
            self.corpus[i] = self.tokenizer.tokenize(self.corpus[i])
            # 将大写字符转换为小写
            for x in range(len(self.corpus[i])):
                self.corpus[i][x] = self.corpus[i][x].lower()
        # 去除停用词表中的停用词
        for i in self.corpus:
            temp = []
            for j in i:
                if j not in self.stopWords:
                    temp.append(j)
            self.corpusNormalizedAndTokenized.append((temp))


    # 建立位置索引
    def buildPositionalIndex(self):
        notebook_dirctory = os.getcwd()
        fileNo = 0
        file_map = {}
        # 进行文件名排序
        files = natsorted(os.listdir(notebook_dirctory + "\\" + self.folderName))
        # 读取每个文件中的内容，保存到self.corpus中
        for i in files:
            self.readFile(i)
        # 进行正则分词、去停用词，保存到self.corpusNormalizedAndTokenized中
        self.tokenizeAndNormalize()

        for corpus in self.corpusNormalizedAndTokenized:

            for index, word in enumerate(corpus):
                # 提取当前词的词干
                stemmedTerm = self.stemmer.stem(word)
                # 如果该词干已存在索引列表中
                if stemmedTerm in self.positionalIndex:

                    # 频率+1
                    self.positionalIndex[stemmedTerm][0] = self.positionalIndex[stemmedTerm][0] + 1

                    # 如果当前文件在该词干的文件库中，把当前词加入索引列表
                    if fileNo in self.positionalIndex[stemmedTerm][1]:
                        self.positionalIndex[stemmedTerm][1][fileNo].append(index)
                    # 如果当前文件不在该词干的文件库中，把当前词作为索引
                    else:
                        self.positionalIndex[stemmedTerm][1][fileNo] = [index]


                # 如果该词干不在索引列表中
                else:

                    # 初始化词干索引列表
                    self.positionalIndex[stemmedTerm] = []
                    # 词干频率+1
                    self.positionalIndex[stemmedTerm].append(1)
                    # 初始化一个空字典
                    self.positionalIndex[stemmedTerm].append({})
                    # 添加文档编号
                    self.positionalIndex[stemmedTerm][1][fileNo] = [index]

            file_map[fileNo] = notebook_dirctory + "/" + self.folderName + "/" + i
            # 进行下一篇文档的分析
            fileNo += 1