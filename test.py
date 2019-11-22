# import json
#
# training_accuracy_list=[]
# step = 0
# while step<3:
#     if step!=0:
#         with open("training_accuracy.json", 'r') as f:
#             test_accuracy = json.load(f)
#         print(len(test_accuracy), test_accuracy)
#
#     training_accuracy_list.append(float(0.130859375))
#     print(type(float(0.130859375)))
#     print(len(training_accuracy_list), step)
#
#     with open("training_accuracy.json", 'w') as f:
#         json.dump(training_accuracy_list,f)
#     step = step + 1

import numpy as np
import math

def calculate(x1, y1, a):
    input = np.matrix([x1, y1]).T
    k = 1.00001330831455
    dx = np.matrix([-108.653343262303, 72.5145588945597]).T
    am = np.matrix([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    output = dx+k*np.matmul(am, input)
    print("input (x1, y1) is : \n{}".format(input))
    print("dx, dy is: \n{}".format(dx))
    print("a (359.595922793000 / math.pi): {}".format(a))
    print("transfer matrix is: \n{}".format(am))
    print("k is: {}".format(k))
    return output

x = [(5410954.17, 502281.044), (5458695.451, 539846.292), (5446979.505, 544651.762)]
a = 359.595922793000 / math.pi
# a = 0
c = 11
for e in x:
    print("point {}".format(c))
    x1, y1 = e
    output = calculate(x1, y1, a)
    print("output (x2, y2) is: \n{}\n".format(output))
    c = c + 1