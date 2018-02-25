import matplotlib.pyplot as plt
import numpy as np

def visualize(accuracies_train, accuracies_val, losses, name):
    plt.figure()
    for i in range(len(accuracies_train)):
        plt.plot(np.arange(len(accuracies_train[i])), accuracies_train[i], label="Fold {}".format(i))
    plt.title("Plot training accuracies for {} folds".format(len(accuracies_train)))
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("Linear_Classifier/{}/{}/t_error_{}.png".format("logs", name, name))
    plt.close()

    plt.figure()
    for i in range(len(accuracies_val)):
        plt.plot(np.arange(len(accuracies_val[i])), accuracies_val[i], label="Fold {}".format(i))
    plt.title("Plot validation accuracies for {} folds".format(len(accuracies_val)))
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("Linear_Classifier/{}/{}/v_error_{}.png".format("logs", name, name))
    plt.close()

    plt.figure()
    for i in range(len(losses)):
        plt.semilogy(np.arange(len(losses[i])), losses[i], label="Fold {}".format(i))
    plt.title("Plot loss for {} folds".format(len(losses)))
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("Linear_Classifier/{}/{}/loss_{}.png".format("logs", name, name))
    plt.close()
