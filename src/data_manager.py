from numpy import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model, datasets, model_selection


class DataManager:
    """
    Data Manager job is to load entire data.
    Data Manager does not care about what kind of data actually exists in dataset.
    We can remove unused data items during specific logistic regression problem.
    We should not use here any patterns that not load specific signs.
    """

    @staticmethod
    def load_data(url: str, miss_first_line: bool, do_convert: bool, split_sign: str) -> list:
        """
        :param url: file to parse url
        :param miss_first_line: miss first line bool value
        :param do_convert: Make conversion to convenient types
        :return: parsed data as data_list
        """
        try:
            file = open(url)
            if miss_first_line:
                next(file)
            data = []
            for line in file.readlines():
                line_elements = line.strip().split(split_sign)
                data_item = []
                for line_el_index in range(len(line_elements)):
                    item = line_elements[line_el_index]
                    try:
                        if do_convert:
                            item = float(line_elements[line_el_index])
                    except:
                        pass
                    data_item.extend([item])
                data.append(data_item)
        finally:
            file.close()
        return data

    @staticmethod
    def categorize_data(data: ndarray, categorical_mask: list):
        """
        Split the data in to:
        - data_non_categoricals
        - data_categoricals
        Assign numerical values to labeled type values
        :param data: ndarray of data without targets
        :param categorical_mask: list with variable categories
        :return: data_non_categoricals, data_categoricals lists
        """
        enc = LabelEncoder()
        for i in range(0, data.shape[1]):
            if (categorical_mask[i]):
                label_encoder = enc.fit(data[:, i])
                print("Klasy kategorialne:", label_encoder.classes_)
                integer_classes = label_encoder.transform(label_encoder.classes_)
                print("Klasy całkowito-liczbowe:", integer_classes)
                t = label_encoder.transform(data[:, i])
                data[:, i] = t

        mask = ones(data.shape, dtype=bool)
        for i in range(0, data.shape[1]):
            if (categorical_mask[i]):
                mask[:, i] = False

        # non categorical data
        data_non_categoricals = data[:, all(mask, axis=0)]

        # categorical data
        data_categoricals = data[:, ~all(mask, axis=0)]

        hotenc = OneHotEncoder()

        hot_encoder = hotenc.fit(data_categoricals)
        encoded_hot = hot_encoder.transform(data_categoricals)

        new_data = append(data_non_categoricals, encoded_hot.todense(), 1)

        return array(new_data)

    @staticmethod
    def train_test_split(inputs: ndarray, target: ndarray, test_size: float, random_state: int):
        """
        :param inputs:
        :param target:
        :param test_size:
        :param random_state:
        :return: X_train, X_test, y_train, y_test ndarrays.
        """
        return model_selection.train_test_split(inputs, target, test_size=test_size, random_state=random_state)

    @staticmethod
    def data_filter(data: ndarray, filter_sign: str) -> ndarray:
        """
        Removes all rows that contains filter_sign.
        :param filter_sign:
        :return: New ndarray without unneeded data items(rows)
        """
        m, n = shape(data)
        inds_to_remove = []
        for i in range(m):
            for j in range(n):
                if data[i, j].strip() == filter_sign:
                    inds_to_remove.append(i)
                    break
        filtered_data = delete(data, inds_to_remove, 0)
        return array(filtered_data)



