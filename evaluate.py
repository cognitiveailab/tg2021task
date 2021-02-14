#!/usr/bin/env python

import logging
from collections import OrderedDict
from typing import Dict, List

import pandas as pd
import json

from evaluation import ExplanationEvaluate

logging.basicConfig(level=logging.DEBUG)


def process_teacher_gold(teacher_preds: List) -> Dict[str, Dict[str, float]]:
    return {
        pred["qid".lower()]: {
            data["uuid"]: data["relevance"] for data in pred["documents"]
        }
        for pred in teacher_preds
    }


def process_teacher_pred(
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


def evaluate():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold", type=argparse.FileType("r", encoding="UTF-8"), required=True
    )
    parser.add_argument("pred", type=argparse.FileType("r", encoding="UTF-8"))
    args = parser.parse_args()

    preds = process_teacher_pred(args.pred)
    gold_explanations = process_teacher_gold(json.load(args.gold)["rankingProblems"])

    rating_threshold = 0

    ndcg_score = ExplanationEvaluate.mean_average_ndcg(
        gold_explanations, preds, rating_threshold
    )
    logging.info(f"Mean NDCG Score : {ndcg_score}")


if __name__ == "__main__":
    evaluate()
