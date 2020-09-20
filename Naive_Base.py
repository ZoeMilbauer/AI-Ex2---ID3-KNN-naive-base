import utils


# Naive Base algorithm
class Naive_Base:
    # initialize data
    def __init__(self, data):
        self.data = data.data
        self.attributes = data.attributes
        self.values_for_attribute = data.values_for_attribute

    # run naive base algorithm on the given train and test
    def run_naive_base(self, train, test):
        num_of_pos_and_neg_on_train = utils.get_num_of_pos_and_neg_examples(train, "label")
        # get number of positive and negative examples in train
        pos_in_train = num_of_pos_and_neg_on_train["yes\n"][0]
        neg_in_train = num_of_pos_and_neg_on_train["no\n"][1]
        # get probability of positive and negative examples in train
        pos_in_train_prob = pos_in_train / len(train)
        neg_in_train_prob = neg_in_train / len(train)
        # initialize number of positive and negative of every attribute
        num_of_pos_and_neg_of_attr = {attr: {} for attr in self.attributes if attr != "label"}
        # go over values for attributes and calculate number of positive and negative
        for attr in self.attributes:
            if attr != "label":
                # go over every value of attribute
                for value in self.values_for_attribute[attr]:
                    # filter data by attribute value
                    info = train.copy()
                    filtered_data = utils.filter_data_by_attr_value(info, attr, value)
                    # save pos and neg of value in num_of_pos_and_neg_of_attr
                    pos_and_neg = utils.get_num_of_pos_and_neg_examples(filtered_data, "label")
                    num_of_pos_and_neg_of_attr[attr][value] = pos_and_neg
        labels = []
        # go over examples in train and calculate probability
        for example in test:
            pos_prob = 1
            neg_prob = 1
            # go over attributes and calculate probability
            for attr in self.values_for_attribute:
                if attr != "label":
                    # get num of pos examples of value of attribute
                    if "yes\n" in num_of_pos_and_neg_of_attr[attr][example[attr]]:
                        pos = num_of_pos_and_neg_of_attr[attr][example[attr]]["yes\n"][0]
                    else:
                        pos = 0
                    # calculate prob
                    pos_prob *= pos / pos_in_train
                    # get num of neg examples of value of attribute
                    if "no\n" in num_of_pos_and_neg_of_attr[attr][example[attr]]:
                        neg = num_of_pos_and_neg_of_attr[attr][example[attr]]["no\n"][1]
                    else:
                        neg = 0
                    # calculate prob
                    neg_prob *= neg / neg_in_train
            # multiply probability in pr(c)
            pos_prob *= pos_in_train_prob
            neg_prob *= neg_in_train_prob
            # determine label according to probability
            if pos_prob > neg_prob:
                labels.append("yes")
            else:
                labels.append("no")
        return labels