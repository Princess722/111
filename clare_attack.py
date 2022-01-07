import torch

import textattack
import transformers
import csv
from tqdm import tqdm
from textattack import Attacker
from textattack.attack_recipes import CLARE2020

device = torch.device("cpu")


def read_corpus(path, text_label_pair=False):
    with open(path, encoding='utf8') as f:
        examples = list(csv.reader(f, delimiter='\t', quotechar=None))[1:]
        second_text = False if examples[1][2] == '' else True
        for i in range(len(examples)):
            examples[i][0] = int(examples[i][0])
            if not second_text:
                examples[i][2] = None

    # label, text1, text2
    if text_label_pair:
        tmp = list(zip(*examples))
        return tmp[0], list(zip(tmp[1], tmp[2]))
    else:
        return examples


if __name__ == "__main__":
    # load dataset
    data = []
    examples = read_corpus("data/yelp.tsv")
    for idx, example in enumerate(tqdm(examples)):
        data.append((example[1], example[0]))

    dataset = textattack.datasets.Dataset(data)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")

    tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    recipe = CLARE2020.build(model_wrapper)

    attack_args = textattack.AttackArgs(log_to_csv="log.csv",
                                        disable_stdout=True,
                                        num_examples=1)

    attacker = Attacker(recipe, dataset, attack_args)
    results = attacker.attack_dataset()
