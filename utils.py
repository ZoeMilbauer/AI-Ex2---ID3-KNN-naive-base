

# gets attribute and returns the number of positive and negative examples
# for every value
def get_num_of_pos_and_neg_examples(info, attribute):
    result = {}
    # go over examples
    for example in info:
        attribute_val = example[attribute]
        # positive example
        if example["label"] == "yes\n":
            if attribute_val in result:
                result[attribute_val][0] += 1
            else:
                result[attribute_val] = [1, 0]
        # negative example
        else:
            if attribute_val in result:
                result[attribute_val][1] += 1
            else:
                result[attribute_val] = [0, 1]
    return result


# gets data, attribute and value and filter data by the value of attribute
def filter_data_by_attr_value(info, attribute, value):
    filtered_data = []
    for example in info:
        if example[attribute] == value:
            filtered_data.append(example)
    return filtered_data


# gets revaluation labels and test and calculate accuracy
def calculate_accuracy(labels, test):
    correct = 0
    for example, label in zip(test, labels):
        if label:
            if example["label"] == (label + "\n"):
                correct += 1
    acc = correct / len(test)
    return acc
