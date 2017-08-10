
from numpy import *
import operator

class BayesClusterClassifier:

    @staticmethod
    def predict(classes: dict, features: dict, target: ndarray, categories: dict):
        """
        For each test item predicts target class and estimation probability
        :return: Each item predicts.
        """
        features_keys = list(features.keys())
        num_of_items = len(features[features_keys[0]])
        results = []

        for item_index in range(num_of_items):
            item_class_probs = {}
            for class_item in classes:
                item_class_probs[class_item] = 1.0

            for feature in features:
                feat_dist_sum = 0
                dists = {}
                for class_item in classes:
                    # if it is categorical
                    item_feat_val = features[feature][item_index] # 23 or 'married'
                    if categories[feature]:
                        # sometimes class category does not contain category item
                        if item_feat_val not in classes[class_item][feature]:
                            item_class_probs[class_item] *= 1
                        else:
                            class_feat_prob = classes[class_item][feature][item_feat_val]
                            item_class_probs[class_item] *= class_feat_prob
                    else:
                        item_dist = pow(classes[class_item][feature] - features[feature][item_index], 2)
                        item_dist = sqrt(item_dist)
                        dists[class_item] = item_dist
                        feat_dist_sum += item_dist
                if not categories[feature]:
                    for class_item in classes:
                        item_class_probs[class_item] *= round(dists[class_item] / feat_dist_sum, 4)
            probs_sum = sum(list(item_class_probs.values()))
            for class_prob in item_class_probs:
                item_class_probs[class_prob] = round(1 - (item_class_probs[class_prob] / probs_sum), 4)
            result = max(item_class_probs.items(), key=operator.itemgetter(1))[0]
            results.append((target[item_index], (result, item_class_probs[result])))
        return results
