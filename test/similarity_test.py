
from src.similarity import Similarity
import unittest

class SimilarityTest(unittest.TestCase):

    def setUp(self):
        print(0)

    def test_eucl_dist(self):
        a = [1,2,3]
        b = [4,5,6]
        similarity = Similarity()
        dist = similarity.euclidean_distance(a,b)
        print(dist)

