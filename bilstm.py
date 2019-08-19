import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.metrics import r2_score
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import RepeatVector
#from keras.layers import TimeDistributed
from attention_decoder import AttentionDecoder
from keras import optimizers
from sklearn.ensemble import AdaBoostRegressor


trainingdataX=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\TrainSXX.csv")
TrainSXX = trainingdataX.values
trainingdataY=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\TrainSYY.csv")
TrainSYY = trainingdataY.values

validdataX=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\TrainSVX.csv")
TrainSVX = validdataX.values
validdataY=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\TrainSVY.csv")
TrainSVY = validdataY.values


testdataX=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\TestSX.csv")
TestSX = testdataX.values


meandata=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\Mu.csv", header=None)
mu = meandata.values
variancedata=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\C.csv", header=None)
sigma = variancedata.values

dimM=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\M.csv", header=None)
M = dimM.values

dimL=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\L.csv", header=None)
L = dimL.values



print(L)

#time index
Xdim=int(20*L);
Ydim=int(M);
TrainSXX = TrainSXX.reshape(-1, 1, (Xdim))
TrainSYY = TrainSYY.reshape(-1, 1, (Ydim))
TrainSVX = TrainSVX.reshape(-1, 1, (Xdim))
TrainSVY = TrainSVY.reshape(-1, 1, (Ydim))
TestSX = TestSX.reshape(-1, 1, (Xdim))


# define model
model = Sequential()
#model.add(Embedding(1000, len(TrainSXX), input_length=Xdim))
model.add(Bidirectional(LSTM(2000,input_shape=(1, Xdim),use_bias=True, return_sequences=True))) #activation='linear' 
#model.add(Bidirectional(LSTM(50,input_shape=(1, Xdim),activation='exponential',use_bias=True,kernel_initializer='normal', return_sequences=True))) 
#model.add(Bidirectional(LSTM(200,activation='linear',use_bias=True, return_sequences=True))) 
#model.add(Dense(1000, kernel_initializer='normal',activation='linear'))
#model.add(Bidirectional(LSTM(500,input_shape=(1, Xdim),use_bias=True, return_sequences=True,recurrent_dropout=0.15,dropout=0.15)))
#model.add(GRU(100, use_bias=True, return_sequences=True)) #
#model.add(Bidirectional(AttentionDecoder(1500, Ydim),merge_mode='sum')) #150
model.add((AttentionDecoder(1500, Ydim)))
#model.add((AttentionDecoder(100, Ydim)))
model.compile(loss='mse', optimizer='Adam',metrics=['acc'])

#boosted_ann = AdaBoostRegressor(base_estimator= model)

#boosted_ann.fit(TrainSXX[:,0], TrainSYY[:,0])# scale your training data 


history=model.fit(TrainSXX, TrainSYY, epochs=200,batch_size=700,shuffle=True,validation_data=(TrainSVX,TrainSVY ))
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.show()
#pred=boosted_ann.predict(TestSX.ravel())
pred=model.predict(TestSX, batch_size=700)
pred1=pred[:,0];

pred2=model.predict(TrainSXX)
pred3=pred2[:,0];

TrainSY=TrainSYY[:,0];

PT= np.arange(0, M, 2);
PH= np.arange(1, M, 2);
P=np.concatenate((PT,PH));

for I in P:
    pred2=sigma[I]*pred1[:,I]+mu[I];
    pred4=sigma[I]*pred3[:,I]+mu[I];
    trainsy=sigma[I]*TrainSY[:,I]+mu[I];

    


    
    testdataY=pd.read_csv("C:\\Users\\MJAVED\\Desktop\\code_mac\\Code\\Python\\Comparison\\TestY.csv")
    TestY = testdataY.iloc[:,I].values
 

 


    mae=mean_absolute_error(TestY,pred2)
    r2=r2_score(TestY,pred2)
    rms = sqrt(mean_squared_error(TestY,pred2))
    
    mae1=mean_absolute_error(trainsy,pred4)
    r21=r2_score(trainsy,pred4)
    rms1 = sqrt(mean_squared_error(trainsy,pred4))

    print('-----------------------------------------------------')
    print(I)
    print('Testing')
    
    print('MAE=', end=' '); print(mae, end=' ')
    print('RMSE=', end=' '); print(rms, end=' ')
    print('R2=', end=' '); print(r2)
    print('Training')
 
    print('MAE=', end=' '); print(mae1, end=' ')
    print('RMSE=', end=' '); print(rms1, end=' ')
    print('R2=', end=' '); print(r21)
    print('-----------------------------------------------------')
   
    #plt.plot(pred2)
    plt.show()
