from Linear_Classifier.train import n_fold as train_lin_model
from Linear_Classifier.visualize import n_fold as train_lin_model
from Neural_Network.train import n_fold as train_net_model
import json

# Load the models from the model directory
MODEL_DIR = "Models/"
LIN_MODELS = "lin_models.json"
NN_MODELS = "nn_models.json"

if __name__ == '__main__':
    lin_models = json.load(open(MODEL_DIR + LIN_MODELS))
    nn_models = json.load(open(MODEL_DIR + NN_MODELS))

    # Perform bulk training and testing of the winning models
    for model in lin_models:
        print(str(model))
        train_lin_model(lin_models[str(model)], test=True)

    for model in nn_models:
        train_net_model(nn_models[str(model)], test=True)
