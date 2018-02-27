import matplotlib.pyplot as plt
import numpy as np

def visualize(accuracies_train, accuracies_val, losses, name, classes):
    plt.figure()
    for i in range(len(accuracies_train)):
        plt.plot(np.arange(len(accuracies_train[i])), accuracies_train[i], label="{}".format(classes[i]))
    plt.title("Training Accuracy")
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("Figures/{}/t_error_{}.png".format(name, name))
    plt.close()

    plt.figure()
    for i in range(len(accuracies_val)):
        plt.plot(np.arange(len(accuracies_val[i])), accuracies_val[i], label="{}".format(classes[i]))
    plt.title("Validation accuracy")
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("Figures/{}/v_error_{}.png".format(name, name))
    plt.close()

    plt.figure()
    for i in range(len(losses)):
        plt.plot(np.arange(len(losses[i])), losses[i], label="{}".format(classes[i]))
    plt.title("Loss")
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("Figures/{}/loss_{}.png".format(name, name))
    plt.close()
