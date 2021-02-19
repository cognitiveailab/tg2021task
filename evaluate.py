#!/usr/bin/env python3
# NDCG scoring function adapted from: https://github.com/kmbnw/rank_metrics/blob/master/python/ndcg.py

import json
import logging
import warnings
from collections import OrderedDict
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable, **kwargs) -> Iterable:
        return iterable

logging.basicConfig(level=logging.DEBUG)


def process_expert_gold(expert_preds: List) -> Dict[str, Dict[str, float]]:
    return {
        pred["qid".lower()]: {
            data["uuid"]: data["relevance"] for data in pred["documents"]
        }
        for pred in expert_preds
    }


def process_expert_pred(
    filepath_or_buffer: str, sep: str = "\t"
) -> Dict[str, List[str]]:
    df = pd.read_csv(
        filepath_or_buffer, sep=sep, names=("question", "explanation"), dtype=str
    )

    if any(df[field].isnull().all() for field in df.columns):
        raise ValueError(
            "invalid format of the prediction dataset, possibly the wrong separator"
        )

    pred: Dict[str, List[str]] = OrderedDict()

    for id, df_explanations in df.groupby("question"):
        pred[id] = list(
            OrderedDict.fromkeys(df_explanations["explanation"].str.lower())
        )

    return pred


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold", type=argparse.FileType("r", encoding="UTF-8"), required=True
    )
    parser.add_argument("--no-tqdm", action="store_false", dest="tqdm")
    parser.add_argument("pred", type=argparse.FileType("r", encoding="UTF-8"))
    args = parser.parse_args()

    preds = process_expert_pred(args.pred)
    gold_explanations = process_expert_gold(json.load(args.gold)["rankingProblems"])

    rating_threshold = 0

    ndcg_score = mean_average_ndcg(gold_explanations, preds, rating_threshold, args.tqdm)
    print(f"Mean NDCG Score : {ndcg_score}")


def mean_average_ndcg(
    gold: Dict[str, Dict[str, float]],
    predicted: Dict[str, List[str]],
    rating_threshold: int,
    use_tqdm: bool
) -> float:
    """Calculate the Mean Average NDCG

    Args:
        gold (Dict[str, Dict[str, float]]): Gold explanations with Question_ID: [{fact_id_1: score_1}, {fact_id_2}:score]
        predicted (Dict[str, List[str]]): Predicted explanations with Question_ID: [fact_id_1, fact_id_2]
        rating_threshold (int): The threshold rating from gold explanations used to calcuate NDCG

    Returns:
        float: returns Mean Average NDCG score or -1 (if proper gold data is not provided)
    """

    if len(gold) == 0:
        logging.error(
            "Empty gold labels. Please verify if you have provided the correct file"
        )
        return -1

    if use_tqdm:
        mean_average_ndcg = np.average(
            [
                ndcg(
                    gold[q_id],
                    list(OrderedDict.fromkeys(predicted[q_id]))
                    if q_id in predicted
                    else [],
                    rating_threshold,
                )
                for q_id in tqdm(gold, "evaluating")
            ]
        )
    else:
        mean_average_ndcg = np.average(
            [
                ndcg(
                    gold[q_id],
                    list(OrderedDict.fromkeys(predicted[q_id]))
                    if q_id in predicted
                    else [],
                    rating_threshold,
                )
                for q_id in gold
            ]
        )

    return mean_average_ndcg


def ndcg(
    gold: Dict[str, float],
    predicted: List[str],
    rating_threshold: int,
    alternate: bool = True,
) -> float:
    """Calculate NDCG value for individual Question-Explanations Pair

    Args:
        gold (Dict[str, float]): Gold expert ratings
        predicted (List[str]): List of predicted ids
        rating_threshold (int): Threshold of gold ratings to consider for NDCG calcuation
        alternate (bool, optional): True to use the alternate scoring (intended to place more emphasis on relevant results). Defaults to True.

    Raises:
        Exception: If ids are missing from prediction. Raises warning.

    Returns:
        float: NDCG score
    """
    if len(gold) == 0:
        return 1

    # Only consider relevance scores greater than 2
    relevance = np.array(
        [
            gold[f_id] if f_id in gold and gold[f_id] > rating_threshold else 0
            for f_id in predicted
        ]
    )

    missing_ids = [g_id for g_id in gold if g_id not in predicted]

    if len(missing_ids) > 0:
        warnings.warn(
            f"Missing gold ids from prediction. Missing ids will be appended to 10**6 position"
        )
        padded = np.zeros(10 ** 6)
        for index, g_id in enumerate(missing_ids):
            padded[index] = gold[g_id]
        relevance = np.concatenate((relevance, np.flip(padded)), axis=0)

    nranks = len(relevance)

    if relevance is None or len(relevance) < 1:
        return 0.0

    if nranks < 1:
        raise Exception("nranks < 1")

    pad = max(0, nranks - len(relevance))

    # pad could be zero in which case this will no-op
    relevance = np.pad(relevance, (0, pad), "constant")

    # now slice downto nranks
    relevance = relevance[0 : min(nranks, len(relevance))]

    ideal_dcg = idcg(relevance, alternate)
    if ideal_dcg == 0:
        return 0.0

    return dcg(relevance, alternate) / ideal_dcg


def dcg(relevance: np.array, alternate: bool = True) -> float:
    """Calculate discounted cumulative gain.

    Args:
        relevance (np.array): Graded and ordered relevances of the results.
        alternate (bool, optional): True to use the alternate scoring (intended to place more emphasis on relevant results). Defaults to True.

    Returns:
        float: DCG score
    """
    if relevance is None or len(relevance) < 1:
        return 0.0

    p = len(relevance)
    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"

        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, relevance) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return relevance[0] + (relevance[1:] / log2i).sum()


def idcg(relevance: np.array, alternate: bool = True) -> float:
    """Calculate ideal discounted cumulative gain (maximum possible DCG)

    Args:
        relevance (np.array): Graded and ordered relevances of the results
        alternate (bool, optional): True to use the alternate scoring (intended to place more emphasis on relevant results).. Defaults to True.

    Returns:
        float: IDCG Score
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = relevance.copy()
    rel.sort()
    return dcg(rel[::-1], alternate)


if __name__ == "__main__":
    main()
