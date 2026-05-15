import spacy
from datasets import load_dataset
from collections import Counter

class Multi30kDataset:
    def __init__(self, split='train'):
        self.split=split
        self.dataset=load_dataset("bentrevett/multi30k", split=self.split)
        self.spacy_en=spacy.load("en_core_web_sm")
        self.spacy_de=spacy.load("de_core_news_sm")
        self._de_tokenized=None
        self._en_tokenized=None
    
    def _tokenize(self, nlp, sentences, batch_size=512):
        return [
            [tok.text for tok in doc] for doc in nlp.pipe(sentences, batch_size=batch_size)
        ]

    def build_vocab(self, min_freq=3):
        special_tokens={"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}

        self._de_tokenized=self._tokenize(self.spacy_de, self.dataset["de"])
        self._en_tokenized=self._tokenize(self.spacy_en, self.dataset["en"])

        de_freq=Counter(tok for sent in self._de_tokenized for tok in sent)
        en_freq=Counter(tok for sent in self._en_tokenized for tok in sent)
        
        self.src_vocab=dict(special_tokens)
        next_idx_de=len(self.src_vocab)
        for token, freq in de_freq.items():
            if freq>min_freq and token not in self.src_vocab:
                 self.src_vocab[token]=next_idx_de
                 next_idx_de+=1

        self.tgt_vocab=dict(special_tokens)
        next_idx_en=len(self.tgt_vocab)
        for token, freq in en_freq.items():
            if freq>min_freq and token not in self.tgt_vocab:
                self.tgt_vocab[token]=next_idx_en
                next_idx_en+=1
        
    def process_data(self):
        if self._de_tokenized is None or self._en_tokenized is None:
            raise RuntimeError("Call build_vocab() before process_data().")
        
        unk_idx_de=self.src_vocab["<unk>"]   
        sos_idx_de=self.src_vocab["<sos>"]
        eos_idx_de=self.src_vocab["<eos>"]
        self.src_data=[
            [sos_idx_de] + [self.src_vocab.get(tok, unk_idx_de) for tok in sent] + [eos_idx_de] for sent in self._de_tokenized
        ]

        unk_idx_en=self.tgt_vocab["<unk>"]   
        sos_idx_en=self.tgt_vocab["<sos>"]
        eos_idx_en=self.tgt_vocab["<eos>"]
        self.tgt_data=[
            [sos_idx_en]+[self.tgt_vocab.get(tok, unk_idx_en) for tok in sent] + [eos_idx_en] for sent in self._en_tokenized
        ]

        return self.src_data, self.tgt_data
        












