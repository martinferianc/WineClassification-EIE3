from Linear_Classifier.train import n_fold as train_lin_model
from Neural_Network.train import n_fold as train_net_model

if __name__ == '__main__':
    lin_model = {}
    lin_model["LEARNING_RATE"] = 0.01
    lin_model["LOSS"] = "squared_loss"
    lin_model["STOP"] = 0.0000001
    lin_model["REGULARIZER"] = "L2"
    lin_model["REGULARIZATION_PENALTY"] = 0.01
    lin_model["EPOCHS"] = 1
    lin_model["N_BATCHES"] = 5
    lin_model["TEST"] = False
    #train_lin_model(lin_model, test=True)

    net_model = {}
    net_model["LEARNING_RATE"] = 0.001
    net_model["HIDDEN_LAYERS"] = 3
    net_model["HIDDEN_NEURONS"] = 1
    net_model["REGULARIZER"] = "L2"
    net_model["REGULARIZATION_PENALTY"] = 0.001
    net_model["ACTIVATION"] = "relu"
    net_model["EPOCHS"] = 1
    net_model["N_BATCHES"] = 10
    net_model["DROPOUT"] = 0.8
    net_model["BETA1"]= 0.99
    train_net_model(net_model, test=True)
