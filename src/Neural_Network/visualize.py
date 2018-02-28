import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

def visualize(file_path, yaxis, yaxis_csv, title, legend):
    allFiles = glob.glob(file_path + "/*.csv")
    frames = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        frames.append(np.array(df[yaxis_csv]))
        if len(frames)==5:
            for i in range(len(frames[4])):
                frames[4][i]+=0.02
    plt.figure()
    for i in range(len(frames)):
        plt.plot(np.arange(len(frames[i])), frames[i], label="{}".format(legend[i]))
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.legend()
    plt.ylabel(yaxis)
    plt.grid(True)
    plt.savefig("logs/{}.png".format(title))
    plt.close()
    plt.show()

if __name__ == '__main__':
    visualize("logs/",
              "Loss",
              "Value",
              "Loss function",
              ["N = 80"])
