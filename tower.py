import os
import pickle

import data_provider
import torch
from biaffine import XLMRobertaForBiaffineParsing
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import XLMRobertaConfig, XLMRobertaTokenizer


class TowerParser:
    def __init__(self, tower_model, device="cpu"):
        self.tower_model = None
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.device = torch.device(device)
        self.load_parser(tower_model)

    def load_parser(self, tower_model):
        if self.tower_model:
            if self.tower_model != tower_model:
                self.tower_model = None
                del self.model
                del self.deps
                self.load_model(tower_model)
        else:
            self.load_model(tower_model)

    def load_model(self, tower_model):
        self.tower_model = tower_model
        or_deps = pickle.load(open(os.path.join(tower_model, "deps.pkl"), "rb"))
        self.deps = {or_deps[k]: k for k in or_deps}
        config = XLMRobertaConfig.from_pretrained(tower_model)
        self.model = XLMRobertaForBiaffineParsing.from_pretrained(
            tower_model, config=config
        )
        self.model.to(self.device)
        self.model.eval()

    def parse(
        self,
        lang,
        sentences,
        batch_size=1,
        max_length=256,
        max_word_len=140,
    ) -> list[list[tuple[int, str, int, str]]]:
        dataset_arcs = []
        dataset_rels = []

        dataset = data_provider.featurize_sents(
            sentences,
            self.tokenizer,
            lang,
            max_length=max_length,
        )

        sampler = SequentialSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        for batch in tqdm(
            loader, desc="Parsing (in batches of " + str(batch_size) + ")"
        ):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                b_outputs = self.model(batch)

            rel_scores = b_outputs[0]
            arc_scores = b_outputs[1]

            arc_preds = arc_scores.argmax(-1)
            if len(arc_preds.shape) == 1:
                arc_preds = arc_preds.unsqueeze(0)

            rel_preds = rel_scores.argmax(-1)
            rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

            lengths = batch[3]
            for i in range(len(rel_preds)):
                arcs = arc_preds[i][: lengths[i]]
                dataset_arcs.append(arcs)

                rels = rel_preds[i][: lengths[i]]
                dataset_rels.append(rels)

        parses = []
        for i in range(len(sentences)):
            sent_parse = []
            for j in range(len(sentences[i])):
                index = j + 1
                token = sentences[i][j]
                governor = dataset_arcs[i][j].item()
                relation = self.deps[dataset_rels[i][j].item()]

                sent_parse.append((index, token, governor, relation))
            parses.append(sent_parse)

        return parses
