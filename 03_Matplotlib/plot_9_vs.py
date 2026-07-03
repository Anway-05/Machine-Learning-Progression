import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from matplotlib.animation import FuncAnimation


def animate(i):
    df=pd.read_csv("data.csv")
    x=df["x_value"]
    c1=df["total_1"]
    c2=df["total_2"]
    
    plt.cla()

    plt.plot(x,c1,label="Channel 1")
    plt.plot(x,c2,label="Channel 2")

    plt.legend(loc="upper left")

plt.tight_layout()
anim = FuncAnimation(plt.gcf(), animate, interval=1000, cache_frame_data=False)

plt.show()
