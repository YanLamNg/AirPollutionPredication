import pandas as pd
import numpy as np

STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]

def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in data[feature].unique():
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data

def readData():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    sta = "Changping"
    df_temp = pd.read_csv("RawData/PRSA_Data_{0}_20130301-20170228.csv".format(sta),
                     usecols = ["year","PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"])


    df_temp = mapping(df_temp, "wd")
    cutoff = df_temp[df_temp.loc[:, "year"] == 2017].index

    df1 = df1.append(df_temp.head(cutoff[0]))
    df2 = df2.append(df_temp.tail(df_temp.shape[0]-cutoff[0]))

    df1 = df1.dropna()
    df2 = df2.dropna()
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    return df1, df2

if __name__ == '__main__':
    df3, df4 = readData()
    df5 = df3.loc[:, ["TEMP", "PRES"]]
    print(df4)
