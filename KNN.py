

# calculate and returns hamming distance between two given strings
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length.")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


# KNN algorithm
class KNN:
    # initialize data
    def __init__(self, data):
        self.data = data.data
        self.attributes = data.attributes
        self.values_for_attribute = data.values_for_attribute

    # gets data and return list of data as string, and list of classification
    def train_to_string(self, info):
        data_as_string = []
        classification = []
        # go over examples and save string and classification
        for example in info:
            example_string = ""
            for attr in self.values_for_attribute:
                if attr == "label":
                    classification.append(example[attr])
                else:
                    example_string += example[attr]
            data_as_string.append(example_string)
        return data_as_string, classification

    # gets data and return list of data as string
    def test_to_string(self, info):
        data_as_string = []
        # go over examples and save string
        for example in info:
            example_string = ""
            for attr in self.values_for_attribute:
                if attr != "label":
                    example_string += example[attr]
            data_as_string.append(example_string)
        return data_as_string

    # run knn algorithm on given train and test, k = 5
    def run_knn(self, train, test):
        # get train and test as strings
        train_as_string, classification = self.train_to_string(train)
        test_as_string = self.test_to_string(test)
        labels = []
        # go over examples of test and classify data
        for example in test_as_string:
            hamming_dis = []
            # go over train examples and calculate hamming distance between
            # every train example and test example
            for train_example in train_as_string:
                hamming_dis.append(hamming_distance(example, train_example))
            # find 5 nearest neighbors
            nearest_neigh = sorted(range(len(hamming_dis)), key=lambda x: hamming_dis[x])[:5]
            pos = 0
            neg = 0
            # find top classification
            for i in nearest_neigh:
                if classification[i] == "yes\n":
                    pos += 1
                else:
                    neg += 1
            if pos > neg:
                labels.append("yes")
            else:
                labels.append("no")
        return labels