import unittest

from evaluate import mean_average_ndcg


class EvaluationTest(unittest.TestCase):
    def test_mean_ndcg(self):
        gold = {"q_id_1": {"id1": 6, "id2": 4, "id5": 10}}
        predicted = {"q_id_1": ["id3", "id2", "id1"]}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 0.08623187912144512)

    def test_perfect_ndcg(self):
        gold = {"q_id_2": {"id1": 10, "id2": 10, "id3": 10}}
        predicted = {"q_id_2": ["id3", "id2", "id1"]}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 1.0)
        predicted = {"q_id_2": ["id2", "id3", "id1"]}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 1.0)
        predicted = {"q_id_2": ["id1", "id2", "id3"]}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 1.0)

    def test_repeated_entries(self):
        gold = {"q_id_2": {"id1": 10, "id2": 10, "id3": 10}}
        predicted = {"q_id_2": ["id3", "id2"]}
        ndcg_score_1 = mean_average_ndcg(gold, predicted, 0, 0)
        predicted = {"q_id_2": ["id3", "id2", "id2"]}
        ndcg_score_2 = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score_1, ndcg_score_2)

    def test_mean_ndcg_missing_prediction(self):
        gold = {"q_id_1": {"id1": 6, "id2": 4, "id5": 10}}
        predicted = {}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 0.05161325042429895)
        # Note: NDCG will never be zero in our evalation script. Since we append the missing predictions at the end

    def test_empty_gold(self):
        gold = {}
        predicted = {}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, -1)

    def test_mean_ndcg_missing_gold(self):
        gold = {"q_id_2": {"id1": 10, "id2": 10, "id3": 10}}
        predicted = {"q_id_1": ["id3", "id2", "id1"], "q_id_2": ["id3", "id2", "id1"]}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 1.0)

        gold = {"q_id_2": {"id1": 10, "id2": 10, "id3": 10}}
        predicted = {"q_id_1": ["id3", "id2", "id1"]}
        ndcg_score = mean_average_ndcg(gold, predicted, 0, 0)
        self.assertAlmostEqual(ndcg_score, 0.07063348642991646)
