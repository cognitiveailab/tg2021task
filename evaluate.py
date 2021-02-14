#!/usr/bin/env python

from collections import OrderedDict
from typing import Dict, List

import click
import pandas as pd
import ujson as json
from loguru import logger

from evaluation import ExplanationEvaluate


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
        pred[id.lower()] = list(
            OrderedDict.fromkeys(df_explanations["explanation"].str.lower())
        )

    return pred


@click.command()
@click.option("--prediction", help="Prediction file")
@click.option("--gold", help="Gold teacher ratings")
def evaluate(prediction, gold):
    preds = process_teacher_pred(prediction)
    with open(gold) as f:
        gold_explanations = process_teacher_gold(json.load(f)["rankingProblems"])

    rating_threshold = 0

    ndcg_score = ExplanationEvaluate.mean_average_ndcg(
        gold_explanations, preds, rating_threshold
    )
    logger.success(f"Mean NDCG Score : {ndcg_score}")


if __name__ == "__main__":
    evaluate()
