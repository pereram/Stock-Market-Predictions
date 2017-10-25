import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def load_data(file_name):
    df=pd.read_csv(file_name, index_col=0)
    #print(df.head())
    return df
    
def printing_unique_tickers(data_frame):
    tickers=df['symbol'].unique()
    print('\nThe all tickers:',tickers)
    print('\nThe number of tickers:',tickers.shape[0])
    
    
def creat_ticker_df(data_frame,ticker_name):
    
    df=data_frame
    df=df[df['symbol']==ticker_name]
    df.drop(['open','high','low','volume', 'symbol'],1,inplace=True)
    df.rename(columns={'close':'{}_close'.format(ticker_name)}, inplace=True)
    df.fillna(0,inplace=True)
    return df


def creat_predict_seq(data_frame,sequence_length):
    df=data_frame
    for index in range(sequence_length):
        df['{}'.format(index+1)]=df.iloc[:,0].shift(-(index+1))
        # creating 51 columns by shifting data by 1.First 50 for X and 51st for y     
    df=df[:-50]# drop last 50 NAN values created due to negative shift
    return df

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    normalised_data=np.array(normalised_data) # converting a numpy array
    return normalised_data

def train_test_split(data_frame,train_prec):
    data=data_frame
    data=data.values # converting data frame into a numpy array
    data=normalise_windows(data)

    row = round(train_prec * data.shape[0])
    train = data[:int(row), :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = data[int(row):, :-1]
    y_test = data[int(row):, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train,y_train,X_test,y_test
    
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

df=load_data('prices-split-adjusted.csv')    
#tickers=df['symbol'].unique()
#print('The all tickers:',tickers)
printing_unique_tickers(df)
ticker_name='AAPL' 

df=creat_ticker_df(df,ticker_name)  
seq_len= 50
df=creat_predict_seq(df,seq_len)
train_prec=0.75

X_train,y_train,X_test,y_test=train_test_split(df,train_prec)
model = build_model([1, 50, 100, 1])

epochs  = 1

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

predicted = predict_point_by_point(model, X_test)
plot_results(predicted,y_test)


print("The score:",model.evaluate(X_test,y_test,batch_size=1))




'''
correlation=df.corr()
k=10
cols=correlation.nlargest(n=k,columns='symbol')['symbol'].index
correlation_zoom=correlation.loc[cols,cols]
plt.subplots(figsize=(12,9))
sns.heatmap(correlation_zoom,annot=True,square=True,xticklabels=cols.values,
            yticklabels=cols.values)
    
'''   
    

