from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter

@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.
    Args:
        filename: Name of the file containing XML markup for labeled alignments
    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    f = open(filename, 'r')
    file_as_string = f.read().replace('&', '&amp;')
    root = ET.fromstring(file_as_string)
    sentence_pairs = list()
    alignments = list()

    for sentence in root:
        source = sentence[0].text.split(' ')
        target = sentence[1].text.split(' ')
        sentence_pair = SentencePair(source, target)
        sentence_pairs.append(sentence_pair)

        alignment_args = []
        for i in [2, 3]:
            arg = list()
            if sentence[i].text is not None:
                for element in sentence[i].text.split(' '):
                    integers = list()
                    for pair in element.split('-'):
                        integers.append(int(pair))
                    arg.append(tuple(integers))
            alignment_args.append(arg)
        alignment = LabeledAlignment(*alignment_args)
        alignments.append(alignment)
    return sentence_pairs, alignments

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.
    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language
    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    Tip:
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary
    """
    source = Counter()
    target = Counter()
    for pair in sentence_pairs:
        source.update(pair.source)
        target.update(pair.target)
        # source[pair.source] += 1
        # target[pair.target] += 1
    dicts = []
    for counter in [source, target]:
        counter_dict = dict()
        new_counter = counter.most_common(freq_cutoff)
        for num, pair in enumerate(new_counter):
            counter_dict[pair[0]] = num
        dicts.append(counter_dict)
    return tuple(dicts)


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language
    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    def tokenize(words, dictionary):
        indexes = list()
        for word in words:
            if word in dictionary:
                indexes.append(dictionary[word])
        if len(words) != len(indexes):
            return None
        else:
            return np.array(indexes)

    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        s = tokenize(pair.source, source_dict)
        t = tokenize(pair.target, target_dict)
        if s is not None and t is not None:
            tokenized_sentence_pairs.append(TokenizedSentencePair(s, t))
    return tokenized_sentence_pairs


