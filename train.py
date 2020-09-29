import mnist_loader
import network

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)

    # print("sample training data")
    # image = np.reshape(training_data[2][0], (28, 28))
    # plt.imshow(image)
    # plt.show()
    # print(image)

    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
    net.SGD(training_data[:1000], 30, 10, 0.5,
            lmbda=5.0,
            evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            early_stopping_n=10)

    net.save("trained.out")