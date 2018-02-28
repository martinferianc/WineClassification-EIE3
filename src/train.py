from Linear_Classifier.train import n_fold as train_lin_model
from Neural_Network.train import n_fold as train_net_model
from visualize import visualize
import json
import os

# Load the models from the model directory
MODEL_DIR = "Models/"
LIN_MODELS = "lin_models.json"
NN_MODELS = "nn_models.json"

# Specify the name of the test
TEST = "N_Neurons"
if __name__ == '__main__':
    
    # Load the models to be trained and tested
    lin_models = json.load(open(MODEL_DIR + LIN_MODELS))
    nn_models = json.load(open(MODEL_DIR + NN_MODELS))

    if not os.path.isdir("Figures/"+TEST):
        os.makedirs("Figures/"+TEST)

    accuracies_val = []
    accuracies_train = []
    losses = []

    final_accuracies = []
    indices = [int(model) for model in lin_models]
    indices.sort()
    for model in indices:
        final_accuracy, accuracy_train, accuracy_val, loss = train_lin_model(lin_models[str(model)], test=False)
        accuracies_val.append(accuracy_val)
        accuracies_train.append(accuracy_train)
        losses.append(loss)
        final_accuracies.append(final_accuracy)

    visualize(accuracies_train, accuracies_val, losses, "Elastic_Net", ["Penalty = 0.1", "Penalty = 0.01", "Penalty = 0.001","Penalty = 0.0001"])

    indices = [int(model) for model in nn_models]
    indices.sort()
    for model in indices:
        train_net_model(nn_models[str(model)], test=True)
