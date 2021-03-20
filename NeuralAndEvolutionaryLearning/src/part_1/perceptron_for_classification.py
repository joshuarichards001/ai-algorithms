

# A instance within the data (represents a line in the data) has a class and a set of values
class Instance:
    def __init__(self, classification, features):
        self.classification = classification
        self.predicted_outcome = 0
        self.learning_rate = 0.01
        self.features = features  # numpy array of features
        self.weights = None  # numpy array of weights for each feature


# Parses a file and turns it into a list of instances
def file_parser(file_name):
    list_instances = list()
    with open(file_name) as f:
        lines = [line.rstrip() for line in f]

    for line in lines[1:]:
        instance_value_list = line.split()
        temp = Instance(int(instance_value_list[len(instance_value_list)-1]), [0] * len(instance_value_list))
        temp.features[0] = 1  # Bias
        for i in range(len(instance_value_list)-1):
            temp.features[i+1] = float(instance_value_list[i])
        temp.weights = [0] * len(temp.features)
        list_instances.append(temp)

    return list_instances


# Classifies the given instance
def perceptron_training(inst):
    total = sum(i * j for i, j in zip(inst.features, inst.weights))
    inst.predicted_outcome = 1 if total > 0 else 0
    for i in range(1, len(inst.weights)):
        inst.weights[i] += inst.learning_rate * (inst.classification - inst.predicted_outcome) * inst.features[i]
    inst.weights[0] += inst.learning_rate * (inst.classification - inst.predicted_outcome)


# Main code
if __name__ == '__main__':
    instances = file_parser("dataset")  # First file

    for _ in range(200):
        for instance in instances:
            perceptron_training(instance)

    count = 0
    for instance in instances:
        print("Actual Class:", instance.classification, "Predicted Class:", instance.predicted_outcome)
        if instance.predicted_outcome == instance.classification:
            count += 1
    print("\nWins: ", count, "Total: ", len(instances))
    print(count/len(instances)*100, "%  is the percentage classified correctly")
