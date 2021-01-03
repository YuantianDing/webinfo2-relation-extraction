
import spacy
import networkx as nx
from typing import List
import re

def clear_hyphen(s):
    return s.replace("--", "$hyphen$").replace("-", "").replace("$hyphen$", "--")

# dep_entities = { "nsubj", "dobj", "pobj", "nsubjpass" }
stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

dep_tree_nlp = spacy.load("en_core_web_sm")

class DepTree:
    def __init__(self, sentence: str):
        """Initialize a dependency tree.

        Args:
            sentence (str): input sentence.
        """
        sentence = preprocessing(sentence)
        self.doc = dep_tree_nlp(sentence) # 使用 spacy 提取依存树
        edges = []
        for token in self.doc:
            edges.append((token.head.i, token.i))
        self.graph = nx.Graph(edges)  # 使用 networkx 构建一张无向图
    
    def shortest_path(self, src: str, dest: str) -> List[str]:
        """Get the shorest path in the dependency graph.

        Args:
            src (str): source vertex(entity)
            dest (str): destination vertex(entity)

        Returns:
            List[str]: shortest path
        """
        src_pos = self.search(src)
        dest_pos = self.search(dest)
        # 调用 networkx 的最短路径算法
        try:
            pos_path = nx.shortest_path(self.graph, source=src_pos, target=dest_pos)
            return [ self.doc[i].text for i in pos_path if self.doc[i].text not in stopwords]
        except nx.NetworkXNoPath:
            return []

    def search(self, entity: str) -> int:
        entity = clear_hyphen(entity).lower()
        for token in self.doc:
            if token.text == entity:
                return token.i
        
