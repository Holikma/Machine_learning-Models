from pydataset import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd

def randompoints():
    x = np.array(rnd.choices(range(10), k = 15))
    y = np.array(rnd.choices(range(10), k = 15))
    return x,y

def LM(x,y):
    n = np.size(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*mean_x*mean_y
    SS_xx = np.sum(x*x) - n*mean_x*mean_y
    #regression coeficients
    b1 = SS_xy / SS_xx
    b0 = mean_y - b1*mean_x
    print(f"n:{n}\nmean_x:{mean_x}\nmean_y:{mean_y}\nSS_xy:{SS_xy}\nSS_xx:{SS_xx}\nb1:{b1}\nb0:{b0}")
    return (b0, b1)

def plot_regression_line(x,y,b):
    plt.scatter(x,y, s= 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color="g")
    plt.show()

def main():
    
    x,y = randompoints()
    plot_regression_line(x,y, LM(x,y))

    

if __name__ == "__main__":
    main()