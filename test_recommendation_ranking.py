import unittest
from unittest.mock import patch

from score_svd_recommender import get_top_recommendations


class RecommendationRankingTest(unittest.TestCase):
    def test_content_similarity_outweighs_svd_after_a_view(self):
        class Model:
            def predict(self, uid, iid):
                return type('Prediction', (), {'est': 2.0 if iid == 'unrelated' else 1.0})()

        with patch('score_svd_recommender._load_model', return_value=Model()), \
             patch('score_svd_recommender._catalog_isbns', return_value=['related', 'unrelated']), \
             patch('score_svd_recommender._content_scores', return_value={'RELATED': 0.9, 'UNRELATED': 0.2}), \
             patch('score_svd_recommender._year_scores', return_value={}), \
             patch('score_svd_recommender._popularity_scores', return_value={}), \
             patch('score_svd_recommender._availability_counts', return_value={}):
            results = get_top_recommendations('user', [])

        self.assertEqual(['RELATED', 'UNRELATED'], [item['isbn'] for item in results])


if __name__ == '__main__':
    unittest.main()
