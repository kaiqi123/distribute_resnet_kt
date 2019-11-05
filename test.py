
import json

training_accuracy_list=[]
step = 0
while step<3:
    if step!=0:
        with open("training_accuracy.json", 'r') as f:
            test_accuracy = json.load(f)
        print(len(test_accuracy), test_accuracy)

    training_accuracy_list.append(float(0.130859375))
    print(type(float(0.130859375)))
    print(len(training_accuracy_list), step)

    with open("training_accuracy.json", 'w') as f:
        json.dump(training_accuracy_list,f)
    step = step + 1