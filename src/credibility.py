

"""
Współczynnik FP = FP/N, gdzie N = TN+FP
 Swoistość = 1–współczynnik FP = TN/N
 Czułość = TP/P, gdzie P = TP+FN
 Precyzja = TP/(TP+FP) - trafne predykcje pozytywne w stosunku do sumy trafnych predykcji pozytywnych oraz fałszywych predykcji pozytywnych.
 Trafność = (TP+TN)/(P+N)
 F-score = precyzja×trafność - ilość odpowiedzi prawdziwych

* get trafne predykcje


Trafne predykcje pozytywne (TP) Trafne predykcje negatywne (TN)
Fałszywe Fałszywe predykcje pozytywne (FP) Fałszywe predykcje negatywne (FN)
"""
class Credibility:
    """
    Computes credibility for the algorithm results,
    Create method for estimating the results for the multiclass problems,

    """
    def __init__(self, results: list, task_classes: list, class_props):
        """
        :param results: List of tuples. Each tuple contain estimated and target value.
        """
        self.task_classes = task_classes
        self.results = results
        self.class_props = class_props
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        # self.get_error_mat()

    def get_predictions(self):
        return self.TP/len(self.results)

    # TO REFACTOR
    def get_specificity(self):
        N = self.TN + self.FP
        if N == 0:
            return 0
        return self.TN/N

    def get_sensitivity(self):
        P = self.TP + self.FN
        if P == 0:
            return 0
        return self.TP / P

    """
    def get_precision(self):
        value = (self.TP + self.FP)
        if value == 0:
            return 0
        return self.TP / (self.TP + self.FP)
    """

    def get_accuracy(self):
        P = self.TP + self.FN
        N = self.TN + self.FP
        return (self.TP + self.TN) / (P + N)

    def get_f_score(self):
        if self.get_accuracy() == 0:
            return 0
        return self.get_precision() / self.get_accuracy()

    def get_precision(self):

        # task classes
        classes = {}
        num_of_results = len(self.results)
        for class_item in self.task_classes:
            classes[class_item] = 0
        for item in self.results:
            if item[0] == item[1][0]:
                classes[item[0]] += 1

        for class_item in self.task_classes:
            classes[class_item] = round(classes[class_item] / (len(self.results) * self.class_props[class_item]), 4)
        return classes


