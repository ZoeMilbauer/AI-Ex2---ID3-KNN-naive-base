import math
import utils


# gets number of positive examples, and number of negative examples
# and returns the entropy
def entropy(positive, negative):
    if positive == 0 or negative == 0:
        return 0
    sum = positive + negative
    log1 = math.log2(positive / sum)
    log2 = math.log2(negative / sum)
    return ((-positive / sum) * log1) - ((negative / sum) * log2)


# gets attribute and returns average information entropy
def average_info_entropy(info, attribute, num_of_pos, num_of_neg):
    num_of_pos_and_neg = utils.get_num_of_pos_and_neg_examples(info, attribute)
    result = 0
    # fo over values of attribute and calculate average information entropy
    for value in num_of_pos_and_neg:
        ent = entropy(num_of_pos_and_neg[value][0], num_of_pos_and_neg[value][1])
        result += ((num_of_pos_and_neg[value][0] + num_of_pos_and_neg[value][1])/(num_of_pos + num_of_neg))*ent
    return result


# ID3 algorithm
class ID3:
    # initialize data
    def __init__(self, data):
        self.data = data.data
        self.attributes = data.attributes
        self.values_for_attribute = data.values_for_attribute

    # calculate gain of given data on given attribute
    def gain(self, info, attribute):
        # get positive and negative classification of data
        num_of_pos_and_neg_in_data = utils.get_num_of_pos_and_neg_examples(info, "label")
        num_of_pos = num_of_pos_and_neg_in_data["yes\n"][0]
        num_of_neg = num_of_pos_and_neg_in_data["no\n"][1]
        # calculate entropy of data
        data_entropy = entropy(num_of_pos, num_of_neg)
        # return gain
        return data_entropy - average_info_entropy(info, attribute, num_of_pos, num_of_neg)

    # id3 algorithm : gets data and attributes and build decision tree
    def DTL(self, info, attributes):
        # if there are no attributes (besides can_eat), check common tag on data and return it
        if len(attributes) == 1:
            pos_and_neg = utils.get_num_of_pos_and_neg_examples(info, "label")
            if pos_and_neg["yes\n"][0] > pos_and_neg["no\n"][1]:
                return "yes"
            else:
                return "no"
        tree = {}
        gains = {}
        # find gain of every attribute
        for i in range(0, len(attributes) - 1):
            gains[attributes[i]] = self.gain(info, attributes[i])
        # get maximum gain
        max_gain = max(gains, key=gains.get)
        # get classifications of attribute with max gain
        pos_and_neg = utils.get_num_of_pos_and_neg_examples(info, max_gain)
        # remove attribute from attributes
        attributes.remove(max_gain)
        # go over values for the attribute, and make tree
        for value in self.values_for_attribute[max_gain]:
            # if value not in data, create node with max classification of data
            if value not in pos_and_neg:
                tree[max_gain + "=" + value] = self.DTL(info, ["label"])
            # if all examples are negative, make node of no
            elif pos_and_neg[value][0] == 0:
                tree[max_gain + "=" + value] = "no"
            # if all examples are positive, make node of yes
            elif pos_and_neg[value][1] == 0:
                tree[max_gain + "=" + value] = "yes"
            # else, filter info - get info without attribute, and make tree
            else:
                filtered_info = utils.filter_data_by_attr_value(info, max_gain, value)
                tree[max_gain + "=" + value] = self.DTL(filtered_info, attributes.copy())
        return tree

    # write tree to file
    def write_tree(self, tree, file, num_of_tabs):
        tabs = ""
        for i in range(num_of_tabs):
            tabs += "\t"
        # go over tree
        for key, val in tree.items():
            # if it is a leaf, write
            if isinstance(val, str):
                file.write(tabs + "|" + key + ":" + val + "\n")
            # else, write attribute and write the node tree
            if isinstance(val, dict):
                if num_of_tabs > 0:
                    file.write(tabs + "|" + key + "\n")
                else:
                    file.write(tabs + key + "\n")
                self.write_tree(val, file, num_of_tabs+1)

    # run ID3 algorithm on given data
    def run_id3(self, info):
        attributes = self.attributes.copy()
        tree = self.DTL(info, attributes)
        return tree

    # gets example and a tree and follow decision tree until gets classification
    def follow_decision_tree(self, example, tree):
        # if it is a leaf, return answer
        if isinstance(tree, str):
            return tree
        # go over tree
        for key, value in tree.items():
            splitted_string = key.split("=")
            if example[splitted_string[0]] == splitted_string[1]:
                return self.follow_decision_tree(example, value)

    # gets test and tree and returns classification to test according to tree
    def get_classification_from_tree(self, tree, test):
        labels = []
        for example in test:
            label = self.follow_decision_tree(example, tree)
            labels.append(label)
        return labels
