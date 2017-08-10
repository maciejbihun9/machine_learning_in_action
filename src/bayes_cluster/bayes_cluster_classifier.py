
from numpy import *

class BayesClusterClassifier:
    # get not splited data

    # compute clusters params

    # if category is float then use cluster method

     # get distance between an item and each cluster
     # get prob that an item belongs to given cluster.

    # if category is string type then use bayes method


    @staticmethod
    def predict(classes: dict, features: dict, categories: dict):
        # get feature length
        # all features has to be the same length
        features_keys = list(features.keys())
        num_of_items = len(features_keys[0])

        class_probs = {}
        for class_item in classes:
            class_probs[class_item] = 1.0

        for item_index in range(num_of_items):
            for feature in features:
                feat_dist_sum = 0
                for class_item in classes:
                    # if it is categorical
                    item_feat_val = features[feature][item_index] # 23 or 'married'
                    if categories[feature]:
                        class_feat_prob = classes[class_item][feature][item_feat_val]
                        class_probs[class_item] *= class_feat_prob
                    else:
                        item_dist = pow(classes[class_item][feature] - features[feature][item_index], 2)
                        item_dist = sqrt(item_dist)
                        class_probs[class_item] = item_dist
                        feat_dist_sum += item_dist
                if not categories[feature]:
                    for class_item in classes:
                        class_probs[class_item] *= round(class_probs[class_item] / feat_dist_sum, 4)
        return class_probs
