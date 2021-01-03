
import spacy
import networkx as nx
from typing import List
import re

def clear_hyphen(s):
    return s.replace("--", "$hyphen$").replace("-", "").replace("$hyphen$", "--")

dep_tree_nlp = spacy.load("en_core_web_sm")
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

class DepTree:
    def __init__(self, sentence: str):
        sentence = clear_hyphen(sentence)
        self.doc = dep_tree_nlp(sentence)
        edges = []
        for token in self.doc:
            edges.append((token.head.i, token.i))
        self.graph = nx.Graph(edges)
    
    # def entities(self) -> List[str]:
    #     return [ token.text for token in self.doc if token.dep_ in dep_entities and token.text not in stopwords]
    
    def shortest_path(self, src: str, dest: str) -> List[str]:
        src_pos = self.search(src)
        dest_pos = self.search(dest)
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
        
