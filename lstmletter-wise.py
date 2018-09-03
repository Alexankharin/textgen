'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
#from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import numpy as np
from keras import regularizers
from keras import backend as K
#import random
#import sys
#import io
import time

import matplotlib.pyplot as plt
from keras.utils import to_categorical

def distmetrics(y_true, y_pred):
    return K.mean(np.sum((y_true-y_pred)**2, axis=0))

def removefrequentxy(sequences,predict, wordprocessed):
    wordfreq={}
    maxfreq=0.001
    for i in range(len(predict)):
        wordfreq[predict[i]]=wordfreq.get(predict[i],0)+1
        if wordfreq[predict[i]]>maxfreq*len(predict):
            predict[i]=None
            sequences[i]=None
    sequences = [x for x in sequences if x != None]
    predict = [x for x in predict if x != None]
    return sequences, predict

def xyprepare(sequences, predict):
    encoded=[]
    encodedx=np.zeros((len(sequences), len(sequences[0]), len(encoding)))
    seqnum=len(sequences)
    seqlen=len(sequences[0])
    dims= len(encoding)
    for i in range(seqnum):
        for l in range(seqlen):
            encodedx[i,l,encoding[sequences[i][l]]]=1
    #x=np.zeros((seqnum,seqlen,dims))
    if onehot==1:
        for word in predict:
            encoded.append(encoding[word])
        y = to_categorical(np.array(encoded),num_classes=dims)
    return encodedx, y


def sampling(lengthofsample=20, initialstr='Ах, не говорите мне про Австрию! Я', ):
    initial=initialstr.lower()
    initwordprocessed=list(initial)
    sequence=[None]
    for i in range(lengthofsample):
        sequence[0]=initwordprocessed
        predi=initwordprocessed[-1]
        xtest, _ = xyprepare(sequence, predi)
        pred=model.predict(xtest)
        if onehot==1:
            #print (pred)
            wordpred=decoding[np.argmax(pred)]
        initwordprocessed.append(wordpred)
    return initwordprocessed
    
start_time = time.time()

handle=open('lt1.txt', 'r')
Li=handle.read().lower()
wordprocessed=list(Li)
#wordstagged=[]
#=letters
#ru_model = KeyedVectors.load_word2vec_format('ru.vec', binary=False)
#print ("model loaded", (time.time() - start_time), "seconds")
#wordprocessed=[]
#for word in words:
#    if (word in ru_model):
#        wordprocessed.append(word)
 
onehot=1

if onehot==1:
    counter=set(wordprocessed)
    encoding={}
    j=0
    for it in counter:
        encoding[it]=j
        j+=1
    
    decoding={}
    for k,v in encoding.items():
        decoding[v]=k    


        
numberwordsinp=30
batchsize=100000
dimsvec=len(encoding)
epochs=1

   

sequences=[]
predict=[]
for i in range(len(wordprocessed)-numberwordsinp):
    sequences.append(wordprocessed[i:i+numberwordsinp])
    predict.append(wordprocessed[i+numberwordsinp])

#sequences, predict = removefrequentxy(sequences,predict, wordprocessed)


print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(None, dimsvec) ,return_sequences=True))
model.add(Dropout(rate=0.8))
#model.add(BatchNormalization())
#model.add(Dense(300, activation='elu'))
#model.add(BatchNormalization())
model.add(LSTM(128))
#model.add(Dropout(rate=0.8))
model.add(BatchNormalization())
model.add(Dense(300, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.8))
#model.add(BatchNormalization())
model.add(Dense(300, activation='relu'))
model.add(Dropout(rate=0.8))
model.add(BatchNormalization())
optimizer = Adam(lr=0.005)
if onehot==1:
    model.add(Dense(len(encoding), activation='softmax'))#, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

else:
    model.add(Dense(dimsvec, activation='tanh'))#, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
    model.compile(loss='cosine_proximity', optimizer=optimizer, metrics=[distmetrics])


#model.load_weights('nightmodel_elus3epoch')


#print (len(sequences), '  ' , len(predict))
#K.set_value(model.optimizer.lr, 0.001)
loss=[]
for i in range (int(len(sequences)/batchsize)):
    batchseq, batchpred=sequences[i*batchsize:i*batchsize+batchsize], predict[i*batchsize:i*batchsize+batchsize]
    xba, yba= xyprepare(batchseq, batchpred )
    hist=model.fit(xba, yba,
              batch_size=512,
              epochs=epochs,
    callbacks=None)
    loss.append(hist.history['loss'])
    print (i+1, '  th iteration from  ',  int(len(sequences)/batchsize))
toplot=[]
for obj in loss:
    for num in obj:
        toplot.append(num)        
plt.plot(toplot)
model.save('nightmodel_elus3epoch')
print("Saved model to disk")
generated=str().join(sampling(20))
