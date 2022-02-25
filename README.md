# cs6910_assignment1
This repository contains the assignments completed as part of a Deep learning course taught by Prof. Mitesh Khapra.

# Question 1
The code for question 1 can be accessed [here](https://github.com/Kartik0611/cs6910_assignment1/blob/main/question1.py) The program, reads the data from keras.datasets, picks one example from each class and logs the same to wandb.

# Question 2 - 6 
The problem statement is to build and train a Feed Forward Neural Network from scratch using primary Numpy package in Python.

The code base now has the following features:

1. Forward and backward propagation are hard coded using Matrix operations. 
2. A neural network class 
3. The optimisers, activations , loss , intialisers and helper are defined separately for ease of use 
4. A colab notebook containing the entire code to train and validate the model from scratch. [Check](https://colab.research.google.com/drive/1RyCw0a5rMMBtIQ1e2Dmbh3bAYJCMXRpM?authuser=2#scrollTo=5NdX6JmMUzpP)(You will have to upload the python files in order to use the colab. 

# Hyperparameters

It is defined in start.py

```
sweep_config = {
  "name": "Random Sweep", #(or) Bayesian Sweep (or) Grid search
  "method": "random", #(or) bayes (or) grid
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}
```



# Code Execution

Define the hyperparamenters in start.py
```
  python start.py
  python train.py 

```
 
