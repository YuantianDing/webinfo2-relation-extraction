import pandas as pd
from typing import List, Iterable, Iterator, Tuple
from simpletransformers.classification import ClassificationModel
from simpletransformers.ner import NERModel
from src.dependency import DepTree

relation_list = [
    "Component-Whole",
    "Other",
    "Instrument-Agency",
    "Member-Collection",
    "Cause-Effect",
    "Entity-Destination",
    "Content-Container",
    "Message-Topic",
    "Product-Producer",
    "Entity-Origin"
]

class ClassifyModel:
    def __init__(self, path: str):
        self.path = path
        try:
            self.model = ClassificationModel("distilbert", path, use_cuda=False)
        except:
            self.model = ClassificationModel("distilbert", "distilbert-base-uncased", num_labels=len(relation_list), use_cuda=False)
    
    def train(self, train_data: Iterator[Tuple[str, str]]):
        train_df = pd.DataFrame(train_data)
        train.columns = ["text", "labels"]

        self.model.train_model(train_df, output_dir=self.path)
    
    def predict(self, data: Iterable[str]) -> List[str]:
        predictions, _ = self.model.predict(list(data))
        return [ relation_list[i] for i in predictions]

custom_labels = ["O", "E"]

def get_entity(raw):
    value = []
    for d in raw:
        [(word, l)] = list(d.items())
        value.append((max(entry[1] - entry[0] for entry in l), word))
        value.sort()
    return [ w for _, w in value[-2:] ]

class DependencyModel:
    
    import nltk

    def __init__(self, ner_path: str, deps_path: str):
        self.ner_path = ner_path
        self.deps_path = deps_path
        try:
            self.ner_model = NERModel("distilbert", ner_path, use_cuda=False)
        except:
            self.ner_model = NERModel("distilbert", "distilbert-base-uncased",
                labels=custom_labels, use_cuda=False)
        self.deps_model = ClassifyModel(deps_path)
    
    def train_ner(self, ner_data: Iterator[Tuple[str, str, str]] ):
        train_data = []
        tokenizer = nltk.RegexpTokenizer(r"[A-Za-z']+")

        for i, (sentence, e1, e2) in ner_data:
            for word in tokenizer.tokenize(sentence):
                label = "O"
                if word.lower() == e1.lower() or word.lower() == e2.lower():
                    label = "E"
                train_data.append((i, word, label))
        
        train_data = pd.DataFrame(train_data)
        train_data.columns = ["sentence_id", "words", "labels"]
        self.ner_model.train(train_data, output_dir=self.ner_path)

    def train_deps(self, train_data: Iterator[Tuple[str, Tuple[str, str, str]]]):
        classify_data: List[str, str] = []

        for sentence, (relation, e1, e2) in train_data:
            tree = DepTree(sentence)
            classify_data.append((" ".join(tree.shortest_path(e1,e2)), relation_list.index(relation)))
        self.deps_model.train(train_data)
    
    def train(self, train_data: Iterator[Tuple[str, Tuple[str, str, str]]]):
        self.train_ner(self, ((sent, e1, e2) for sent, (rela, e1, e2) in train_data) )
        self.train_deps(self, train_data)
    
    def predict(self, data: Iterable[str]) -> List[Tuple[str, Tuple[str, str]]]:
        data = list(data)
        predictions, raws = self.ner_model.predict(data)

        predict_data = []
        entities = []
        for raw, sentence in zip(raws, data):
            [e1, e2] = get_entity(raw)
            tree = DepTree(sentence)
            entities.append((e1, e2))
            predict_data.append(" ".join(tree.shortest_path(e1,e2)))
        return list(zip(self.deps_model.predict(predict_data), entities))
