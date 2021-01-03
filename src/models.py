import pandas as pd
from typing import List, Iterable, Iterator, Tuple
from simpletransformers.classification import ClassificationModel
from simpletransformers.ner import NERModel
from src.dependency import DepTree
import nltk

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
        """Create/Load a new Model

        Args:
            path (str): if this path exists, then load the model, otherwise it will create a new model at the path.
        """
        self.path = path
        try:
            self.model = ClassificationModel("distilbert", path, use_cuda=False)
        except:
            self.model = ClassificationModel("distilbert", "distilbert-base-uncased", num_labels=len(relation_list), use_cuda=False)
    
    def train(self, train_data: Iterator[Tuple[str, str]]):
        """ Train the model. 

        Args:
            train_data (Iterator[Tuple[str, str]]): An Iterator of the pairs of input sentence and the corresponding label.
        """
        data = [ (sent, relation_list.index(rela)) for sent, rela in train_data]
        train_df = pd.DataFrame(data)
        train_df.columns = ["text", "labels"]

        self.model.train_model(train_df, output_dir=self.path)
    
    def predict(self, data: Iterable[str]) -> List[str]:
        """Use the Model to predict the label of sentence.

        Args:
            data (Iterable[str]): iterable of input sentence string.

        Returns:
            List[str]: list of labels.
        """
        predictions, _ = self.model.predict(list(data))
        return [ relation_list[i] for i in predictions]

custom_labels = ["O", "E"]

def get_entity(raw) -> List[str]:
    """Get two best entity from raw format

    Args:
        raw (raw output from simpletransformer): 

    Returns:
        List[str]: two entities.
    """
    value = []
    for d in raw:
        [(word, l)] = list(d.items())
        value.append((max(entry[1] - entry[0] for entry in l), word))
        value.sort()
    return [ w for _, w in value[-2:] ]

class DependencyModel:
    

    def __init__(self, ner_path: str, deps_path: str):
        """Create/Load a new DependencyModel

        Args:
            ner_path (str): directory of NER model. (if not exists, create a new model)
            deps_path (str): directory of classification model.
        """
        self.ner_path = ner_path
        self.deps_path = deps_path
        try:
            self.ner_model = NERModel("distilbert", ner_path, use_cuda=False)
        except:
            self.ner_model = NERModel("distilbert", "distilbert-base-uncased",
                labels=custom_labels, use_cuda=False)
        self.deps_model = ClassifyModel(deps_path)
    
    def train_ner(self, ner_data: Iterator[Tuple[str, str, str]] ):
        """Train the NER model

        Args:
            ner_data (Iterator[Tuple[str, str, str]]): iterator of tuple of (sentence, entity1, entity2).
        """

        train_data = []
        tokenizer = nltk.RegexpTokenizer(r"[A-Za-z']+")

        for i, (sentence, e1, e2) in enumerate(ner_data):
            for word in tokenizer.tokenize(sentence):
                label = "O"
                if word.lower() == e1.lower() or word.lower() == e2.lower():
                    label = "E"
                train_data.append((i, word, label))
        
        train_data = pd.DataFrame(train_data)
        train_data.columns = ["sentence_id", "words", "labels"]
        self.ner_model.train(train_data, output_dir=self.ner_path)

    def train_deps(self, train_data: Iterator[Tuple[str, Tuple[str, str, str]]]):
        """Train the dependency tree path classification model.

        Args:
            train_data (Iterator[Tuple[str, Tuple[str, str, str]]]): iterator of (sentence, (relation, entity1, entity2))
        """
        classify_data: List[str, str] = []

        for sentence, (relation, e1, e2) in train_data:
            tree = DepTree(sentence)
            classify_data.append((" ".join(tree.shortest_path(e1,e2)), relation_list.index(relation)))
        self.deps_model.train(train_data)
    
    def train(self, train_data: Iterator[Tuple[str, Tuple[str, str, str]]]):
        """Train both the NER model AND the classification model.

        Args:
            train_data (Iterator[Tuple[str, Tuple[str, str, str]]]): iterator of (sentence, (relation, entity1, entity2))
        """
        train_data = list(train_data)
        self.train_ner([ (sent, e1, e2) for sent, (rela, e1, e2) in train_data ])
        self.train_deps(train_data)
    
    def predict(self, data: Iterable[str]) -> List[Tuple[str, Tuple[str, str]]]:
        """Predict the relation extracted from input sentences.

        Args:
            data (Iterable[str]): list of input sentences.

        Returns:
            List[Tuple[str, Tuple[str, str]]]: list of (relation, (entity1, entity2))
        """
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
