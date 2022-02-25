import numpy as np

def meanSquaredErrorLoss( actual, pred):
        error = np.mean((actual - pred) ** 2)
        return error

def crossEntropyLoss( actual, pred):
    error = [-actual[i] * np.log(pred[i]) for i in range(len(pred))]
    error = np.mean(error)
    return error

def accuracy( actual, pred, batch_size):
        actual_label = []
        pred_label = []
        cnt = 0
        i = 0 
        while i < batch_size:
            actual_label.append(np.argmax(actual[:, i]))
            pred_label.append(np.argmax(pred[:, i]))
            if actual_label[i] == pred_label[i]:
                cnt += 1
            i += 1
        value = cnt / batch_size
        return value, actual_label, pred_label
        
def L2RegularisationLoss(weights, weight_decay):
    ALPHA = weight_decay
    return ALPHA * np.sum(
        [
            np.linalg.norm(weights[str(i + 1)]) ** 2
            for i in range(len(weights))
        ]
    )
