import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


(trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()

N_train_full = trainOut.shape[0]
N_train = int(0.9*N_train_full)
N_validation = int(0.1 * trainOut.shape[0])
N_test = testOut.shape[0]


idx  = np.random.choice(trainOut.shape[0], N_train_full, replace=False)
idx2 = np.random.choice(testOut.shape[0], N_test, replace=False)

trainInFull = trainIn[idx, :]
trainOutFull = trainOut[idx]

trainIn = trainInFull[:N_train,:]
trainOut = trainOutFull[:N_train]

validIn = trainInFull[N_train:, :]
validOut = trainOutFull[N_train:]    

testIn = testIn[idx2, :]
testOut = testOut[idx2]


sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER", "HE"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": [ 'TANH',  'SIGMOID', 'RELU']
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
