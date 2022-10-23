"""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_barplot():
    ## save barplot
    df = pd.read_csv("results/Scores.csv")
    labels = ['SVR                                RF']
    SVR = np.round_(df["SVR"].tolist(), decimals=2)
    RF = np.round_(df["RF"].tolist(), decimals=2)
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 12))
    rects1 = ax.bar(x - width / 2, SVR, width, label='SVR')
    rects2 = ax.bar(x + width / 2, RF, width, label='RF')
    ax.set_ylabel('Test R2', fontweight='bold', fontsize=25)
    ax.set_title('Model comparison', fontweight='bold', fontsize=25)
    ax.set_xticks(x, labels, fontsize=25)
    ax.legend()
    ax.bar_label(rects1, padding=3, fontsize=25)
    ax.bar_label(rects2, padding=3, fontsize=25)
    fig.tight_layout()
    plt.savefig("results/scores.png")

save_barplot()"""