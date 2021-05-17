from Posi_Index_Builder import Posi_Index_Builder
from TFIDF_Builder import TFIDF_Builder


if __name__ == '__main__':
    # 文件预处理和建立索引
    Index_Build = Posi_Index_Builder("docs")
    Index_Build.buildPositionalIndex()

    # 计算语料库中词语的TF、IDF、TF-IDF
    vectorBuilder = TFIDF_Builder(Index_Build.corpus,
                                  Index_Build.corpusNormalizedAndTokenized,
                                  Index_Build.positionalIndex)
    print('计算每个词在每篇文档中的TF值:')
    print(vectorBuilder.get_TF())
    print('计算每一个词的IDF值:')
    print(vectorBuilder.get_IDF())
    print('计算每个词在每篇文档中的TF-IDF值:')
    print(vectorBuilder.get_TFIDF())

    # 输入查询语句
    query = input('输入查询语句或者cancel退出')
    while query.lower() != 'cancel':
        # 把输入的语句与每一篇文档进行比对
        print(vectorBuilder.findSimilarityBetweenQueryAndAllDocs(query))
        # 继续查询
        query = input('输入查询语句或者cancel退出')