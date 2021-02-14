#!/usr/bin/env python3

import sys
import warnings
from collections import OrderedDict
from functools import partial
from typing import List, Dict, Callable, Optional

import pandas as pd


class IncompletePredictionWarning(UserWarning):
    pass


def load_gold(filepath_or_buffer: str, sep: str = '\t') -> Dict[str, List[str]]:
    df = pd.read_csv(filepath_or_buffer, sep=sep, dtype=str)

    df = df[df['flags'].str.lower().isin(('success', 'ready'))]
    df = df[['QuestionID', 'explanation']]
    df.dropna(inplace=True)

    df['QuestionID'] = df['QuestionID'].str.lower()
    df['explanation'] = df['explanation'].str.lower()

    gold: Dict[str, List[str]] = OrderedDict()

    for _, row in df.iterrows():
        gold[row['QuestionID']] = [uid for e in row['explanation'].split()
                                   for uid, _ in (e.split('|', 1),)]

    return gold


def load_pred(filepath_or_buffer: str, sep: str = '\t') -> Dict[str, List[str]]:
    df = pd.read_csv(filepath_or_buffer, sep=sep, names=('question', 'explanation'), dtype=str)

    if any(df[field].isnull().all() for field in df.columns):
        raise ValueError('invalid format of the prediction dataset, possibly the wrong separator')

    pred: Dict[str, List[str]] = OrderedDict()

    for id, df_explanations in df.groupby('question'):
        pred[id.lower()] = list(OrderedDict.fromkeys(df_explanations['explanation'].str.lower()))

    return pred


def average_precision_score(gold: List[str], pred: List[str],
                            callback: Optional[Callable[[int, int], None]] = None) -> float:
    if not gold or not pred:
        return 0.

    correct = 0

    ap = 0.

    true = set(gold)

    for rank, element in enumerate(pred):
        if element in true:
            correct += 1

            if callable(callback):
                callback(correct, rank)

            ap += correct / (rank + 1.)

            true.remove(element)

    if true:
        warnings.warn('pred is missing gold: ' + ', '.join(true), IncompletePredictionWarning)

    return ap / len(gold)


def mean_average_precision_score(golds: Dict[str, List[str]], preds: Dict[str, List[str]],
                                 callback: Optional[Callable[[str, float], None]] = None) -> float:
    if not golds or not preds:
        return 0.

    sum_ap = 0.

    for id, gold in golds.items():
        if id in preds:
            pred = preds[id]

            score = average_precision_score(gold, pred)

            if callable(callback):
                callback(id, score)

            sum_ap += score

    return sum_ap / len(golds)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    parser.add_argument('pred', type=argparse.FileType('r', encoding='UTF-8'))
    args = parser.parse_args()

    gold, pred = load_gold(args.gold), load_pred(args.pred)

    print('{:d} gold questions, {:d} predicted questions'.format(len(gold), len(pred)),
          file=sys.stderr)

    # callback is optional, here it is used to print intermediate results to STDERR
    mean_ap = mean_average_precision_score(
        gold, pred, callback=partial(print, file=sys.stderr)
    )

    print('MAP: ', mean_ap)


if '__main__' == __name__:
    main()
