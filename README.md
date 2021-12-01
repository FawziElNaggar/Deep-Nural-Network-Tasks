# Deep-Nural-Network-Tasks
Create ANN and CNN to train the model

### 1.1) Assignment 1 Day_1:
Design your own simple ANN, (one perceptron with one input layer and one output neuron). Use the data points listed in the adjacent Table as your training data. Assume the activation function is sigmoid and assume there is no bias for simplicity (b=0). Test your design using different iteration numbers.

### 1.2)Assignment 2 Day_1: 
Modify the above-designed code to implement a multi-layer perceptron, MLP (an ANN with one input layer, one hidden layer and one output layer) for the same data points above. Assume sigmoid activation function and there is no bias for simplicity (b=0). Test your approach using different iteration numbers and different number of nodes for the hidden layer (e.g., 4, 8, and 16)

### 1.3)Assignment 3 Day_1: 
Use the Keras library (tensorflow.keras) to build different ANNs using different numbers of hidden layers (shallow: 1 hidden, output layer, deeper: two hidden layers with 12 and , 8) nodes respectively, and more deep: three hidden layers with 32, 16, 8 nodes respectively). Use the provided diabetic data sets (here) to train and test your design. Use the ReLU activation for the hidden layers and the sigmoid activation for the output neuron, loss='binary_crossentropy', optimizer='adam’, metrics=['accuracy’], epochs = 150.

### 1.4)Assignment 4 Day_1: 
Redo assignment #3 using 80% of the data for training and 20% of the data for testing. Also, plot the training accuracy and loss curves for your designed networks

### 2.1)Assignment 1 Day_2: 
Design your own deep NN to classify the CIFAR10 images (you can download from keras.dataset) into one of the 10 classes.
❑ Investigate the use of different architectures (different layers, learning rate, optimizers, loss function).
❑ Note: you will need to flatten the image and use it as your input vector


### 2.2) Assignment 2 Day_2: 
Design your deep convolutional neural network (CNN) to classify the CIFAR10 images into one of the 10 classes.
❑ Invistage the use of different architectures (different layers, kernel sizes, pooling, learning rate, optimizers, loss function).

### 2.3) Assignmetn 3 Day_2: 
Repeat Assignment #1 and #2 using MNIST dataset. 
Note that you will need to convert the training labels into categorical using one hot encoding using to_categorical() function




