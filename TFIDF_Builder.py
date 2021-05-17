import math
import os
import nltk
from nltk.stem import PorterStemmer
from natsort import natsorted
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


class TFIDF_Builder:
    def __init__(self, rawData, data, positionalIndex):
        # 元数据(未处理的语料库)
        self.rawData = rawData
        # 预处理后的语料库
        self.data = data
        # 词频（term frequency，TF)
        self.tf = []
        # 逆向文件频率（inverse document frequency，IDF）
        self.idf = []
        # tf-idf值，即TF与IDF的乘积
        self.TFIDF = []
        # 词干索引列表
        self.positionalIndex = positionalIndex

    # 计算每个词在每篇文档中的词频（TF值）
    def get_TF(self):
        self.tf = defaultdict(list)
        count = 0
        # 初始化每一个词的TF值都为0
        for i in self.rawData:
            for x in i:
                self.tf[x] = [0 for f in range(len(self.rawData))]
        # 统计词频，保留4位小数
        for i in range(len(self.rawData)):
            for x in self.rawData[i]:
                count = self.rawData[i].count(x)
                self.tf[x][i] = round(float(count), 4)
        return self.tf

    # 计算每一个词的IDF值
    def get_IDF(self):
        # 初始化列表
        self.idf = defaultdict(list)
        for i in range(len(self.rawData)):
            for x in self.rawData[i]:
                # 把“包含该词语的文件的数目”初始化为0
                numberOfDocsWithTerm = 0
                # 若该词包含在该文件中，则numnumberOfDocsWithTerm+1
                for f in self.rawData:
                    if x in f:
                        numberOfDocsWithTerm += 1
                # 计算IDF,用总文件数目除以包含该词语的文件的数目，再取对数，结果保留4位小数
                self.idf[x] = round(math.log10(float(len(self.data) / numberOfDocsWithTerm)), 4)

        return self.idf

    # 计算每个词在每篇文档中的TF-IDF值
    def get_TFIDF(self):
        self.TFIDF = defaultdict(float)
        # 初始化每一个词的TF-IDF值都为0
        for i in self.rawData:
            for x in i:
                self.TFIDF[x] = [0 for f in range(len(self.rawData))]
        # 计算TF-IDF值：把每个词的TF值与IDF相乘，结果保留4为小数
        for i in range(len(self.rawData)):
            for x in self.rawData[i]:
                if len(x) <= 1:
                    continue
                self.TFIDF[x][i] = round(self.tf[x][i] * self.idf[x], 4)

        return self.TFIDF

    # 计算查询语句与文档之间的相似度
    def findSimilarity(self, query, doc):
        similarity = 0

        vector_query = []
        vector_doc = []

        # 对查询语句和文档进行分词
        temp_query = word_tokenize(query)
        temp_doc = word_tokenize(doc)
        # 去停用词
        sw = set(stopwords.words('english'))
        tokenizedAndNormalizedQuery = {word for word in temp_query if word not in sw}
        tokenizedAndNormalizedDoc = {word for word in temp_doc if word not in sw}
        # vector包含查询语句和文档中的所有词
        vector = tokenizedAndNormalizedDoc.union(tokenizedAndNormalizedQuery)
        for word in vector:
            # 如果当前词在文档中
            if word in tokenizedAndNormalizedDoc:
                vector_query.append(1)
            # 如果当前词不在文档中
            else:
                vector_query.append(0)
            # 如果当前词在查询语句中
            if word in tokenizedAndNormalizedQuery:
                vector_doc.append(1)
            # 如果当前词不在查询语句中
            else:
                vector_doc.append(0)
        # 取交集，若某词同时在查询语句和文档中，则为1，否则为0；把所有词取交集的结果相加得到similarity
        for i in range(len(vector)):
            similarity += vector_doc[i] * vector_query[i]

        # 文档的词数
        # print(sum(vector_doc))
        # 查询语句的词数
        # print(sum(vector_query))

        # 计算最终相似度：
        answer = similarity / float((sum(vector_doc) * sum(vector_query)) ** 0.5)
        return answer

    def findSimilarityBetweenQueryAndAllDocs(self, query):
        allSimilarity = []
        for i in range(len(self.rawData)):
            # 对每一篇文档，计算查询语句与当前文档的相似度
            allSimilarity.append((i + 1, self.findSimilarity(query, " ".join(self.rawData[i]))))
        # 按相似度从大到小排序
        allSimilarity.sort(key=lambda x: x[-1], reverse=True)
        # 返回top5的文档编号及相似度
        return allSimilarity[:5]