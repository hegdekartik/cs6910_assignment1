import numpy as np
import time
from loss import *
import wandb

class optimizers():
    def __init__(self):
        pass

    def sgd(self, epochs, length_dataset, learning_rate, X_train,FFNN,weight_decay=0):
            
            trainingloss = []
            trainingaccuracy = []
            validationaccuracy = []
            
            num_layers = len(FFNN.layers)

            X_train = X_train[:, :length_dataset]
            Y_train = FFNN.Y_train[:, :length_dataset]

            for epoch in range(epochs):
                start_time = time.time()
                
                idx = np.random.shuffle(np.arange(length_dataset))
                X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
                Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)
                
                LOSS = []

                
                deltaw = [
                    np.zeros((FFNN.layers[l + 1], FFNN.layers[l]))
                    for l in range(0, len(FFNN.layers) - 1)
                ]
                deltab = [
                    np.zeros((FFNN.layers[l + 1], 1))
                    for l in range(0, len(FFNN.layers) - 1)
                ]

                for i in range(length_dataset):

                    Y, H, A = FFNN.fp.forwardPropagate(
                        X_train[:, i].reshape(FFNN.img_flattened_size, 1),
                        FFNN.weights,
                        FFNN.biases,
                        FFNN
                    )
                    grad_weights, grad_biases = FFNN.bp.backPropagate(
                        Y, H, A, Y_train[:, i].reshape(FFNN.num_classes, 1),FFNN
                    )
                    deltaw = [
                        grad_weights[num_layers - 2 - i] for i in range(num_layers - 1)
                    ]
                    deltab = [
                        grad_biases[num_layers - 2 - i] for i in range(num_layers - 1)
                    ]


                    if FFNN.loss_function == "MSE":
                        LOSS.append(meanSquaredErrorLoss(
                                FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                            )
                            + L2RegularisationLoss(FFNN.weights,weight_decay)
                            )
                    elif FFNN.loss_function == "CROSS":
                        LOSS.append(
                            crossEntropyLoss(
                                FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                            )
                            + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )

                    
                    FFNN.weights = {
                        str(i + 1): (FFNN.weights[str(i + 1)] - learning_rate * deltaw[i])
                        for i in range(len(FFNN.weights))
                    }
                    FFNN.biases = {
                        str(i + 1): (FFNN.biases[str(i + 1)] - learning_rate * deltab[i])
                        for i in range(len(FFNN.biases))
                    }

                elapsed = time.time() - start_time
                
                Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
                
                trainingloss.append(np.mean(LOSS))
                trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
                validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])
                
                print(
                            "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                            % (
                                epoch,
                                trainingloss[epoch],
                                trainingaccuracy[epoch],
                                validationaccuracy[epoch],
                                elapsed,
                                FFNN.learning_rate,
                            )
                        )

                wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch, })
            # data = [[epoch, loss[epoch]] for epoch in range(epochs)]
            # table = wandb.Table(data=data, columns = ["Epoch", "Loss"])
            # wandb.log({'loss':wandb.plot.line(table, "Epoch", "Loss", title="Loss vs Epoch Line Plot")})
            return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


        
    def sgdMiniBatch(self, epochs,length_dataset, batch_size, learning_rate, X_train,FFNN,weight_decay = 0):

        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        

        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(FFNN.layers)
        num_points_seen = 0


        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)
            
            LOSS = []
            #Y_pred = []
            
            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]

            for i in range(length_dataset):
                
                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), FFNN.weights, FFNN.biases,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                
                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:
                    
                    
                    FFNN.weights = {str(i+1):(FFNN.weights[str(i+1)] - learning_rate*deltaw[i]/batch_size) for i in range(len(FFNN.weights))} 
                    FFNN.biases = {str(i+1):(FFNN.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(FFNN.biases))}
                    
                    #resetting gradient updates
                    deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
                    deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
            
            elapsed = time.time() - start_time

            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
            
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred



    def mgd(self, epochs,length_dataset, batch_size, learning_rate,X_train,FFNN, weight_decay = 0):
        GAMMA = 0.9

        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(FFNN.layers)
        prev_v_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        prev_v_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)

            LOSS = []

            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
            

            for i in range(length_dataset):
                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), FFNN.weights, FFNN.biases,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,FFNN.Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
                    
                    FFNN.weights = {str(i+1) : (FFNN.weights[str(i+1)] - v_w[i]) for i in range(len(FFNN.weights))}
                    FFNN.biases = {str(i+1): (FFNN.biases[str(i+1)] - v_b[i]) for i in range(len(FFNN.biases))}

                    prev_v_w = v_w
                    prev_v_b = v_b

                    #resetting gradient updates
                    deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
                    deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })


        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred




    def stochasticNag(self,epochs,length_dataset, learning_rate,X_train,FFNN, weight_decay = 0):
        GAMMA = 0.9

        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        

        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(FFNN.layers)
        
        prev_v_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        prev_v_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)

            LOSS = []
            #Y_pred = []  
            
            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
            
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(FFNN.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(FFNN.layers)-1)]
                        
            for i in range(length_dataset):
                winter = {str(i+1) : FFNN.weights[str(i+1)] - v_w[i] for i in range(0, len(FFNN.layers)-1)}
                binter = {str(i+1) : FFNN.biases[str(i+1)] - v_b[i] for i in range(0, len(FFNN.layers)-1)}
                
                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), winter, binter,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,FFNN.Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)
                
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(FFNN.num_classes,))
                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )

                
                v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
        
                FFNN.weights = {str(i+1):FFNN.weights[str(i+1)] - v_w[i] for i in range(len(FFNN.weights))} 
                FFNN.biases = {str(i+1):FFNN.biases[str(i+1)] - v_b[i] for i in range(len(FFNN.biases))}
                
                prev_v_w = v_w
                prev_v_b = v_b

            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


    def nag(self,epochs,length_dataset, batch_size,learning_rate,X_train,FFNN, weight_decay = 0):
        GAMMA = 0.9

        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        


        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(FFNN.layers)
        
        prev_v_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        prev_v_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)

            LOSS = []
            #Y_pred = []  
            
            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
            
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(FFNN.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(FFNN.layers)-1)]

            for i in range(length_dataset):
                winter = {str(i+1) : FFNN.weights[str(i+1)] - v_w[i] for i in range(0, len(FFNN.layers)-1)}
                binter = {str(i+1) : FFNN.biases[str(i+1)] - v_b[i] for i in range(0, len(FFNN.layers)-1)}
                
                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), winter, binter,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,FFNN.Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(FFNN.num_classes,))
                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )

                
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:                            

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
        
                    FFNN.weights ={str(i+1):FFNN.weights[str(i+1)]  - v_w[i] for i in range(len(FFNN.weights))}
                    FFNN.biases = {str(i+1):FFNN.biases[str(i+1)]  - v_b[i] for i in range(len(FFNN.biases))}
                
                    prev_v_w = v_w
                    prev_v_b = v_b

                    deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
                    deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]


            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred



    def rmsProp(self, epochs,length_dataset, batch_size, learning_rate, X_train,FFNN,weight_decay = 0):


        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(FFNN.layers)
        EPS, BETA = 1e-8, 0.9
        
        v_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        v_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        num_points_seen = 0        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)


            LOSS = []
            #Y_pred = []
                        
            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]

            for i in range(length_dataset):
            
                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), FFNN.weights, FFNN.biases,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,FFNN.Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)
            
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                
                #Y_pred.append(Y.reshape(FFNN.num_classes,))
                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                        
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )

                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:
                
                    v_w = [BETA*v_w[i] + (1-BETA)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA*v_b[i] + (1-BETA)*(deltab[i])**2 for i in range(num_layers - 1)]

                    FFNN.weights = {str(i+1):FFNN.weights[str(i+1)]  - deltaw[i]*(learning_rate/np.sqrt(v_w[i]+EPS)) for i in range(len(FFNN.weights))} 
                    FFNN.biases = {str(i+1):FFNN.biases[str(i+1)]  - deltab[i]*(learning_rate/np.sqrt(v_b[i]+EPS)) for i in range(len(FFNN.biases))}

                    deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
                    deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]

            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred  



    def adam(self, epochs,length_dataset, batch_size, learning_rate,X_train,FFNN, weight_decay = 0):
        
        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        

        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        num_layers = len(FFNN.layers)
        EPS, BETA1, BETA2 = 1e-8, 0.9, 0.99
        
        m_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        m_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        v_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        v_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]        
        
        m_w_hat = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        m_b_hat = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        v_w_hat = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        v_b_hat = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]   
        
        num_points_seen = 0 
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)


            LOSS = []
            #Y_pred = []
            
            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
            
            
            for i in range(length_dataset):
                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), FFNN.weights, FFNN.biases,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,FFNN.Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(FFNN.num_classes,))
                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )

                
                num_points_seen += 1
                ctr = 0
                if int(num_points_seen) % batch_size == 0:
                    ctr += 1
                
                    m_w = [BETA1*m_w[i] + (1-BETA1)*deltaw[i] for i in range(num_layers - 1)]
                    m_b = [BETA1*m_b[i] + (1-BETA1)*deltab[i] for i in range(num_layers - 1)]
                
                    v_w = [BETA2*v_w[i] + (1-BETA2)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA2*v_b[i] + (1-BETA2)*(deltab[i])**2 for i in range(num_layers - 1)]
                    
                    m_w_hat = [m_w[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]
                    m_b_hat = [m_b[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]            
                
                    v_w_hat = [v_w[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    v_b_hat = [v_b[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                
                    FFNN.weights = {str(i+1):FFNN.weights[str(i+1)] - (learning_rate/np.sqrt(v_w[i]+EPS))*m_w_hat[i] for i in range(len(FFNN.weights))} 
                    FFNN.biases = {str(i+1):FFNN.biases[str(i+1)] - (learning_rate/np.sqrt(v_b[i]+EPS))*m_b_hat[i] for i in range(len(FFNN.biases))}

                    deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
                    deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]


            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred



    def nadam(self, epochs,length_dataset, batch_size, learning_rate, X_train,FFNN,weight_decay = 0):

        X_train = X_train[:, :length_dataset]
        Y_train = FFNN.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        num_layers = len(FFNN.layers)
        
        GAMMA, EPS, BETA1, BETA2 = 0.9, 1e-8, 0.9, 0.99

        m_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        m_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        v_w = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        v_b = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]        

        m_w_hat = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        m_b_hat = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
        
        v_w_hat = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
        v_b_hat = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)] 

        num_points_seen = 0 
        
        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(FFNN.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(FFNN.num_classes, length_dataset)

            LOSS = []
            #Y_pred = []

            deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
            deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]

            for i in range(length_dataset):

                Y,H,A = FFNN.fp.forwardPropagate(X_train[:,i].reshape(FFNN.img_flattened_size,1), FFNN.weights, FFNN.biases,FFNN) 
                grad_weights, grad_biases = FFNN.bp.backPropagate(Y,H,A,FFNN.Y_train[:,i].reshape(FFNN.num_classes,1),FFNN)

                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(FFNN.num_classes,))
                if FFNN.loss_function == "MSE":
                    LOSS.append(meanSquaredErrorLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                        )
                elif FFNN.loss_function == "CROSS":
                    LOSS.append(
                        crossEntropyLoss(
                            FFNN.Y_train[:, i].reshape(FFNN.num_classes, 1), Y
                        )
                        + L2RegularisationLoss(FFNN.weights,weight_decay)
                    )

                num_points_seen += 1
                
                if num_points_seen % batch_size == 0:
                    
                    m_w = [BETA1*m_w[i] + (1-BETA1)*deltaw[i] for i in range(num_layers - 1)]
                    m_b = [BETA1*m_b[i] + (1-BETA1)*deltab[i] for i in range(num_layers - 1)]
                    
                    v_w = [BETA2*v_w[i] + (1-BETA2)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA2*v_b[i] + (1-BETA2)*(deltab[i])**2 for i in range(num_layers - 1)]
                    
                    m_w_hat = [m_w[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]
                    m_b_hat = [m_b[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]            
                    
                    v_w_hat = [v_w[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    v_b_hat = [v_b[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    
                    
                    FFNN.weights = {str(i+1):FFNN.weights[str(i+1)] - (learning_rate/(np.sqrt(v_w_hat[i])+EPS))*(BETA1*m_w_hat[i]+ (1-BETA1)*deltaw[i]) for i in range(len(FFNN.weights))} 
                    FFNN.biases = {str(i+1):FFNN.biases[str(i+1)] - (learning_rate/(np.sqrt(v_b_hat[i])+EPS))*(BETA1*m_b_hat[i] + (1-BETA1)*deltab[i]) for i in range(len(FFNN.biases))}

                    deltaw = [np.zeros((FFNN.layers[l+1], FFNN.layers[l])) for l in range(0, len(FFNN.layers)-1)]
                    deltab = [np.zeros((FFNN.layers[l+1], 1)) for l in range(0, len(FFNN.layers)-1)]
                
            elapsed = time.time() - start_time

            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = FFNN.predict(X_train, FFNN.N_train,FFNN)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(FFNN.Y_val, FFNN.predict(FFNN.X_val, FFNN.N_val,FFNN), FFNN.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            FFNN.learning_rate,
                        )
                    )
            wandb.log({'loss':np.mean(LOSS), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
            
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred  