def train():    
    config_defaults = dict(
            max_epochs=10,
            num_hidden_layers=3,
            num_hidden_neurons=32,
            weight_decay=0,
            learning_rate=1e-3,
            optimizer="NADAM",
            batch_size=16,
            activation="SIGMOID",
            initializer="XAVIER",
            loss="MSE",
        )
    configdef = {
            'max_epochs':10,
            'num_hidden_layers':3,
            'num_hidden_neurons':128,
            'weight_decay':0,
            'learning_rate':'1e-3',
            'optimizer':"NADAM",
            'batch_size':32,
            'activation':"SIGMOID",
            'initializer':"XAVIER",
            'loss':"MSE"
            }
        
    #wandb.init(config = config_defaults)
    wandb.init(project='cs6910_assignment1', entity='cs6910_assignment1',config = configdef)
    

    wandb.run.name = "MSE_hl_" + str(wandb.config.num_hidden_layers) + "_hn_" + str(wandb.config.num_hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_lr_" + str(wandb.config.learning_rate) + "_bs_"+str(wandb.config.batch_size) + "_init_" + wandb.config.initializer + "_ep_"+ str(wandb.config.max_epochs)+ "_l2_" + str(wandb.config.weight_decay) 
    CONFIG = wandb.config


    
    #sweep_id = wandb.sweep(sweep_config)
  

    FFNN = FeedForwardNeuralNetwork(
        num_hidden_layers=CONFIG.num_hidden_layers,
        num_hidden_neurons=CONFIG.num_hidden_neurons,
        X_train_raw=trainIn,
        Y_train_raw=trainOut,
        N_train = N_train,
        X_val_raw = validIn,
        Y_val_raw = validOut,
        N_val = N_validation,
        X_test_raw = testIn,
        Y_test_raw = testOut,
        N_test = N_test,
        optimizer = CONFIG.optimizer,
        batch_size = CONFIG.batch_size,
        weight_decay = CONFIG.weight_decay,
        learning_rate = CONFIG.learning_rate,
        max_epochs = CONFIG.max_epochs,
        activation = CONFIG.activation,
        initializer = CONFIG.initializer,
        loss = CONFIG.loss
        )



    training_loss, trainingaccuracy, validationaccuracy, Y_pred_train = FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.learning_rate,FFNN.X_train,FFNN)


wandb.agent(sweep_id, train, count = 100)