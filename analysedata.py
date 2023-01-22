import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb



def analysedata(dataframe, dataval, filepath):
    df=pd.read_csv(filepath)
    datavals=df[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].sum()
    lenval=len(df)
    datavals=datavals/lenval
    fig, ax = plt.subplots()
    # print('this is the dataval',dataval.iloc[0].to_numpy())
    x1=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    x=np.arange(7)

    ax.bar(x-0.2,datavals,0.4,label='your')
    ax.bar(x+0.2, dataval.iloc[0].to_numpy(),0.4,label='average')
    plt.xticks(x,x1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the graph
    plt.show()



