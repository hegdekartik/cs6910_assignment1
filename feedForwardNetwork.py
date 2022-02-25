import wandb 
import time
from helper import *
from initializers import *
from loss import *
from optimizers import *
from activators import *
from forward_propagation import *
from backPropagation import *
from prediction import *

class FeedForwardNeuralNetwork:
    def __init__(
        self, 
        num_hidden_layers, 
        num_hidden_neurons, 
        X_train_raw, 
        Y_train_raw,  
        N_train, 
        X_val_raw, 
        Y_val_raw, 
        N_val,
        X_test_raw, 
        Y_test_raw, 
        N_test,        
        optimizer,
        batch_size,
        weight_decay,
        learning_rate,
        max_epochs,
        activation,
        initializer,
        loss

    ):

        
        self.num_classes = np.max(Y_train_raw) + 1  # NUM_CLASSES
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.output_layer_size = self.num_classes
        self.img_height = X_train_raw.shape[1]
        self.img_width = X_train_raw.shape[2]
        self.img_flattened_size = self.img_height * self.img_width
        self.fp = fp()
        self.bp = bp()
        # self.layers = layers
        self.layers = (
            [self.img_flattened_size]
            + num_hidden_layers * [num_hidden_neurons]
            + [self.output_layer_size]
        )

        self.N_train = N_train
        self.N_val = N_val
        self.N_test = N_test
        

        

        self.X_train = np.transpose(
            X_train_raw.reshape(
                X_train_raw.shape[0], X_train_raw.shape[1] * X_train_raw.shape[2]
            )
        )  # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]
        self.X_test = np.transpose(
            X_test_raw.reshape(
                X_test_raw.shape[0], X_test_raw.shape[1] * X_test_raw.shape[2]
            )
        )  # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]
        self.X_val = np.transpose(
            X_val_raw.reshape(
                X_val_raw.shape[0], X_val_raw.shape[1] * X_val_raw.shape[2]
            )
        )  # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]


        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255
        self.X_val = self.X_val / 255
        
        encoder = OHE()
        num_classes = self.num_classes
        self.Y_train = encoder.oneHotEncode(num_classes,Y_train_raw)  # [NUM_CLASSES X NTRAIN]
        self.Y_val = encoder.oneHotEncode(num_classes,Y_val_raw)
        self.Y_test = encoder.oneHotEncode(num_classes,Y_test_raw)


        self.Activations_dict = {"SIGMOID": sigmoid, "TANH": tanh, "RELU": relu}
        self.DerActivation_dict = {
            "SIGMOID": sigmoid_derivative,
            "TANH": tanh_derivative,
            "RELU": relu_derivative,
        }

        Intializer = Intializers()
        self.Initializer_dict = {
            "XAVIER": Intializer.Xavier_initializer,
            "RANDOM": Intializer.random_initializer,
            "HE": Intializer.He_initializer
        }

        optimizer_functions = optimizers()
        self.Optimizer_dict = {
            "SGD": optimizer_functions.sgdMiniBatch,
            "MGD": optimizer_functions.mgd,
            "NAG": optimizer_functions.nag,
            "RMSPROP": optimizer_functions.rmsProp,
            "ADAM": optimizer_functions.adam,
            "NADAM": optimizer_functions.nadam,
        }
        
        self.activation = self.Activations_dict[activation]
        self.der_activation = self.DerActivation_dict[activation]
        self.optimizer = self.Optimizer_dict[optimizer]
        self.initializer = self.Initializer_dict[initializer]
        self.loss_function = loss
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.predict = prediction()
        self.predict = self.predict.predict
        self.weights, self.biases = self.initializeNeuralNet(self.layers)
        

    def initializeNeuralNet(self, layers):
        weights = {}
        biases = {}
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.initializer(size=[layers[l + 1], layers[l]])
            b = np.zeros((layers[l + 1], 1))
            weights[str(l + 1)] = W
            biases[str(l + 1)] = b
        return weights, biases

    
    
    
    

    