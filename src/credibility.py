
class Credibility:
    """
    Computes classifier methods precision.
    """

    def __init__(self, results: list, task_classes: list, class_props: dict):
        """
        :param results: List of tuples which contains target mapped with estimated target and that target estimation probability.
        :param task_classes: List with possible task target classes.
        :param class_props: Dict with each target class occurrence possibility.
        """
        self.task_classes = task_classes
        self.results = results
        self.class_props = class_props

    def get_precision(self):
        """
        Computes % precision for each task class.
        :return: Dict with % estimation precision for each class.
        """
        classes = {}
        for class_item in self.task_classes:
            classes[class_item] = 0
        for item in self.results:
            if item[0] == item[1][0]:
                classes[item[0]] += 1
        for class_item in self.task_classes:
            classes[class_item] = round(classes[class_item] / (len(self.results) * self.class_props[class_item]), 4)
        return classes


