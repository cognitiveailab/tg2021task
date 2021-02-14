import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from .ndcg import ndcg as ndcg_eval


class ExplanationEvaluate:
    """Evaluation explanation reconstruction"""

    @staticmethod
    def mean_average_ndcg(
        gold: Dict[str, List[Dict[str, float]]],
        predicted: Dict[str, List[str]],
        rating_threshold: int,
    ) -> float:
        """Calculate the Mean Average NDCG score

        Args:
            gold (Dict[str,List[Tuple[float,str]]]): Gold explanations with Question_ID: [{fact_id_1: score_1}, {fact_id_2}:score]
            predicted (Dict[str,List[str]]): Predicted explanations with Question_ID: [fact_id_1, fact_id_2]
        """
        logger.info("Calculating Mean Average NDCG")
        if len(gold) == 0:
            logger.error(
                "Empty gold labels. Please verify if you have provided the correct file"
            )
            return -1

        mean_average_ndcg = np.average(
            [
                ExplanationEvaluate.ndcg(
                    q_id,
                    gold[q_id],
                    list(OrderedDict.fromkeys(predicted[q_id]))
                    if q_id in predicted
                    else [],
                    rating_threshold,
                )
                for q_id in tqdm(gold, "Evaluting NDCG")
            ]
        )
        return mean_average_ndcg

    @staticmethod
    def ndcg(
        q_id: str,
        gold: List[Dict[str, float]],
        predicted: List[Dict[str, float]],
        rating_threshold: int,
    ) -> float:
        """Calculate NDCG of one instance

        Args:
            q_id (str): Question ID
            gold (List[Dict[str, float]]): [{fact_id_1: score_1}, {fact_id_2}:score]
            predicted (List[str]): [fact_id_1, fact_id_2]
        """

        if len(gold) == 0:
            return 1

        # Only consider relevance scores greater than 2
        relevance_scores = np.array(
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
            relevance_scores = np.concatenate(
                (relevance_scores, np.flip(padded)), axis=0
            )
        return ndcg_eval(relevance_scores, len(relevance_scores))

    @staticmethod
    def precision_at_k(
        gold: Dict[str, List[Dict[str, float]]],
        predicted: Dict[str, List[str]],
        k: int,
        rating_threshold: int,
    ) -> float:
        """Calculate the Precision@K

        Args:
            gold (Dict[str,List[Tuple[float,str]]]): Gold explanations with Question_ID: [{fact_id_1: score_1}, {fact_id_2}:score]
            predicted (Dict[str,List[str]]): Predicted explanations with Question_ID: [fact_id_1, fact_id_2]
            k:int the parameter k for the precision
            rating: the rating category to consider for the precision
        """
        logger.info("Calculating Precision@K")

        average_precision = 0
        i = 0

        for q_id in gold:
            if not q_id in predicted:
                logger.warning(f"Missing {q_id} in prediction")
                continue
            predictions = predicted[q_id][:k]
            tp = 0
            tot_facts = 0
            for fact in gold[q_id]:
                if gold[q_id][fact] <= rating_threshold:
                    continue
                if fact in predictions:
                    tp += 1
                tot_facts += 1
            average_precision += tp / (len(predictions))

        average_precision /= len(gold)

        logger.success(f"Precision@{k}: {average_precision}")
