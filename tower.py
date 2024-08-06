import os
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import XLMRobertaConfig, XLMRobertaTokenizerFast

from towerparse.biaffine import XLMRobertaForBiaffineParsing
from towerparse.data_provider import language_specific_preprocessing


class TowerParser:
    def __init__(self, tower_model, device="cpu"):
        self.tower_model = None
        self.tokenizer: XLMRobertaTokenizerFast = (
            XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
        )
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
        lang: str,
        sentences: list[list[str]],
        batch_size=1,
        verbose: bool = False,
    ) -> list[list[tuple[int, str, int, str]]]:
        preprocessed_sentences = [
            language_specific_preprocessing(lang, sent) for sent in sentences
        ]

        if not preprocessed_sentences or not any(preprocessed_sentences):
            return []
        
        batch_encoding = self.tokenizer(
            preprocessed_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        batch_subwords = [
            self.tokenizer.convert_ids_to_tokens(
                input_ids,
                skip_special_tokens=True,
            )
            for input_ids in batch_encoding.input_ids
        ]

        word_start_positions = [
            [i for (i, subword) in enumerate(subwords, 1) if subword.startswith("▁")]
            for subwords in batch_subwords
        ]
        word_start_positions = [
            torch.tensor(wsp + [len(subwords) + 1], dtype=torch.long)
            for wsp, subwords in zip(word_start_positions, batch_subwords)
        ]
        lengths = torch.tensor(
            [wsp[-1] for wsp in word_start_positions], dtype=torch.long
        )
        word_start_positions = pad_sequence(
            word_start_positions, batch_first=True, padding_value=-1
        )

        dataset = TensorDataset(
            batch_encoding.input_ids,
            batch_encoding.attention_mask,
            word_start_positions,
            lengths,
        )

        sampler = SequentialSampler(dataset)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
        )

        dataset_arcs = []
        dataset_rels = []
        for batch in tqdm(
            loader,
            desc="Parsing (in batches of " + str(batch_size) + ")",
            disable=not verbose,
        ):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                rel_scores, arc_scores = self.model(batch)

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
