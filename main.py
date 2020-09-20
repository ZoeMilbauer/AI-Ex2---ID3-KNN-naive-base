import Naive_Base
import KNN
import ID3
import utils


# calculate 5 fold cross validation
def k_fold_cross_validation(data):
    acc_file = open("accuracy.txt", 'w')
    id3 = ID3.ID3(data)
    knn = KNN.KNN(data)
    naive_base = Naive_Base.Naive_Base(data)
    parts = []
    start = 0
    end = int(len(data.data) / 5) + 1
    # part train to 5 parts
    for i in range(5):
        parts.append(data.data[start:end + start])
        if i == 3:
            end -= 1
        start = start + end
    train = [parts[1] + parts[2] + parts[3] + parts[4], parts[0] + parts[2] + parts[3] + parts[4],
             parts[0] + parts[1] + parts[3] + parts[4], parts[0] + parts[1] + parts[2] + parts[4],
             parts[0] + parts[1] + parts[2] + parts[3]]
    average_acc_for_id3 = 0
    average_acc_for_knn = 0
    average_acc_for_naive_base = 0
    # run id3, knn and naive base models and calculate accuracy
    for i in range(5):
        tree = id3.run_id3(train[i])
        labels = id3.get_classification_from_tree(tree, parts[i])
        average_acc_for_id3 += utils.calculate_accuracy(labels, parts[i])
        labels = knn.run_knn(train[i], parts[i])
        average_acc_for_knn += utils.calculate_accuracy(labels, parts[i])
        labels = naive_base.run_naive_base(train[i], parts[i])
        average_acc_for_naive_base += utils.calculate_accuracy(labels, parts[i])
    average_acc_for_id3 /= 5
    average_acc_for_knn /= 5
    average_acc_for_naive_base /= 5
    print("average accuracy for ID3: " + str(average_acc_for_id3))
    print("average accuracy for KNN: " + str(average_acc_for_knn))
    print("average accuracy for Naive Base: " + str(average_acc_for_naive_base))
    acc_file.write("{0:.2f}".format(average_acc_for_id3) + "\t" + "{0:.2f}".format(average_acc_for_knn) + "\t" +
                   "{0:.2f}".format(average_acc_for_naive_base))
    acc_file.close()


# DataReader gets data file and extract data, attributes, labels
class DataReader:
    # read data from file
    def __init__(self, data_file):
        self.data = []
        file = open(data_file, 'r')
        lines = file.readlines()
        # save attributes
        self.attributes = lines[0].split('\t')[:-1]
        self.attributes.append("label")
        self.values_for_attribute = {key: [] for key in self.attributes}
        # go over lines and save data
        for i in range(1, len(lines)):
            example = lines[i].split('\t')
            info = {}
            # save values of current example to dictionary, key is attribute, val is value
            for j in range(len(example)):
                info[self.attributes[j]] = example[j]
                if example[j] not in self.values_for_attribute[self.attributes[j]]:
                    # add value of attribute to list
                    self.values_for_attribute[self.attributes[j]].append(example[j])
            # add values to data
            self.data.append(info)
        for attr in self.attributes:
            values = self.values_for_attribute[attr]
            values.sort()
            self.values_for_attribute[attr] = values


if __name__ == '__main__':
    file_name = "dataset.txt"
    data = DataReader(file_name)
    k_fold_cross_validation(data)
