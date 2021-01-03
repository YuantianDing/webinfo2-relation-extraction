
def pairs(seq):
    it = iter(seq)
    try:
        while True:
            yield (next(it), next(it))
    except StopIteration:
        pass

def every_4(seq):
    it = iter(seq)
    try:
        while True:
            yield (next(it), next(it), next(it), next(it))
    except StopIteration:
        pass

def read_train(filename: str):
    import re
    pat_sentence = r'[0-9]+ "(.*?)"\n'
    pat_relation = r'(.*?)\((.*?),(.*?)\)\n'

    with open(filename) as f:
        for sentence, relation in pairs(f):
            sentence = re.fullmatch(pat_sentence, sentence)[1]
            match = re.fullmatch(pat_relation, relation)
            try:
                relation = match[1]
                src = match[2].split(" ")[-1]
                dest = match[3].split(" ")[-1]

                yield (sentence, (relation, src, dest))
            except Exception as e:
                print(f"At sentence: {sentence}, relation: {relation}")
                raise e

def read_test(filename: str):
    import re
    pat_sentence = r'[0-9]+ "(.*?)"\n'
    
    with open(filename) as f:
        for line in f:
            sentence = re.fullmatch(pat_sentence, line)[1]
            yield sentence

def read_data_full(filename: str):
    import re
    pat_sentence = r'[0-9]+\t"(.*?)"\n'
    pat_relation = r'([^.\(\n]*)'
    pat_e1 = r'\<e1\>(.*?)\<\/e1\>'
    pat_e2 = r'\<e2\>(.*?)\<\/e2\>'
    with open(filename) as f:
        for sentence, relation, _, _ in every_4(f):
            sentence = re.fullmatch(pat_sentence, sentence)[1]
            try:
                e1 = re.search(pat_e1, sentence)[1]
                e2 = re.search(pat_e2, sentence)[1]

                sentence = re.sub(pat_e1, e1, sentence)
                sentence = re.sub(pat_e2, e2, sentence)
                match = re.match(pat_relation, relation)

                relation = match[1]
                src = e1.split(" ")[-1]
                dest = e2.split(" ")[-1]

                yield (sentence, (relation, src, dest))
            except Exception as e:
                print(f"At sentence: {sentence}, relation: {relation}")
                raise e

