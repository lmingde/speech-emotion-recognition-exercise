import librosa
import librosa.display
import os
import numpy as np
import pandas as pd
from time import time
from scipy.stats import pearsonr
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPooling1D,LSTM,Reshape,multiply,dot,Flatten
pd.options.display.max_columns= 999
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from keras.callbacks import EarlyStopping 
from keras.optimizers import Adam


def get_feature(audio_path):
    '''using librosa to extract feature from audio file(.wav)
    Parameters
    audio_path: String, the directory path of audios
    Returns:
    feature dataframe
    '''
    t0 = time()
    df = []
    print('extract feature....')
    audio_list = os.listdir(audio_path)
    i = 0
    for audio in audio_list:
        if i%100==0: print('%dth done'%i)
        i += 1
        if audio[-4:] != '.wav': continue
        this_path = os.path.join(audio_path,audio)
        y,sr=librosa.load(this_path,sr=None)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        chroma=librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
        mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        contrast=librosa.feature.spectral_contrast(y=y_harmonic,sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zrate=librosa.feature.zero_crossing_rate(y_harmonic)
        melspc = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
        features = [chroma,mfccs,cent,contrast,rolloff,zrate,melspc]
        features = np.concatenate(features,axis=0).T 
        
        df.append([audio[:-4],features.astype(np.float64)])

    df = pd.DataFrame(df,columns=['id','features'])
    print('extract feature done in %0.3fs'%(time()-t0))
    return df

# data preprocessing
train_data,test_data = get_feature('./audio_data/train/audio'),get_feature('./audio_data/test/audio/')
train_target = pd.read_csv('./data/train_target.csv',sep='\t',header=None,names=['id','target'])
train_data['id'],test_data['id'] = train_data['id'].apply(pd.to_numeric),test_data['id'].apply(pd.to_numeric)
train_data = pd.merge(train_data,train_target,on='id',how='left')
test_data['target'] = -5


data = pd.concat([train_data,test_data],copy=False)
features = data['features'].values
features = np.rollaxis(np.dstack(features),-1)
shape = features.shape
features = features.reshape((len(features),-1))
from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
features = ssc.fit_transform(features)
features = features.reshape(shape)
features = [features[i] for i in range(len(features))]
data['features'] = features

# self define metric to caculate pearsonr
from keras import backend as K
def pearson_r(y_true, y_pred):
    ''' Customized metrics funtion to cacualte pearsonr score
    Parameters:
    y_ture: the ture value of target
    y_pred: the predict value of target
    Returns:
    pearsonr score 
    '''
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def get_model(shape):
    ''' creata a network model and return
    Parameters:
    shape: tuple, the shape of data which model need to train
    Returns:
    network model
    '''
    window_length = 20

    x_input = Input(shape=(shape[-2],shape[-1]),name='input')
    x_mid = Conv1D(filters=32,kernel_size = window_length,activation='relu')(x_input)
    x_mid = MaxPooling1D(2)(x_mid)
    x_mid = Conv1D(32,kernel_size=window_length,activation='relu')(x_mid)
    x_mid = LSTM(512,dropout=0.1, recurrent_dropout=0.2,activation='relu',return_sequences=True)(x_mid)
    attation = Dense(1)(x_mid)
    attation = Reshape((1,x_mid._keras_shape[1]))(attation)
    attation = Dense(x_mid._keras_shape[1],activation='softmax')(attation)
    output = dot([attation,x_mid],axes=(2,1))
    output = Flatten()(output)
    output = Dense(512)(output)
    output = Dense(512)(output)
    output = Dense(512)(output)
    output = Dense(512)(output)
    output = Dense(128)(output)
    output = Dense(1)(output)
    model = Model(inputs=[x_input],outputs=output)

    model.compile(optimizer=Adam(lr=5e-4),loss=['mse'],metrics=[pearson_r])
    return model



blend_train,blend_test = [],[]
train_id,test_id = None,None
data = data.sort_values(by='id')

#10-KFlod
N = 10
skf = KFold(n_splits=N,shuffle=False,random_state=2333)

train,test = data[data['target']!=-5],data[data['target']==-5]
train_id,y = train.pop('id'),train.pop('target')
X = np.rollaxis(np.dstack(train['features']),-1)

test_id,test_y = test.pop('id'),test.pop('target')
test = np.rollaxis(np.dstack(test['features']),-1)
cv_score,cv_y_pred = [],[]

blend_train.append(np.zeros((len(train),1)) )
blend_test.append(np.zeros((len(test),1)) )
for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
    
    shape = X_train.shape
    model = get_model(shape)
    early_stopping_monitor = EarlyStopping(patience=5)
    fit_pcc,fit_pred,fit_train,fit_score = [],[],[],[]
    for i in range(15):
        model.fit(X_train,y_train,batch_size=64,epochs=1,validation_data=(X_test,y_test))
        y_pred = model.predict(X_test).reshape(-1,)
        evs,mae,mse,r2 = explained_variance_score(y_test,y_pred),mean_absolute_error(y_test,y_pred),mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)
        pcc = pearsonr(y_test,y_pred)[0]
        fit_score.append([evs,mae,mse,r2,pcc])
        fit_pcc.append(pcc)
        fit_train.append(y_pred)
        fit_pred.append(model.predict(test).reshape(-1,))

    best_index = np.argmax(fit_pcc)
    cv_score.append(fit_score[best_index])
    cv_y_pred.append(fit_pred[best_index])

    blend_train[-1][test_in] = np.asarray(fit_train[best_index]).reshape((-1,1))
    blend_test[-1] =blend_test[-1]+ np.asarray(fit_pred[best_index]).reshape((-1,1))/N

    score_df = pd.DataFrame([fit_score[best_index]],columns=['explained_variance_score','mae','mse','r2_score','pearsonr_score'])
    print(score_df)

cv_score = pd.DataFrame(cv_score,columns=['explained_variance_score','mae','mse','r2_score','pearsonr_score'])

# save blend_data 
blend_train.to_csv('./data/dp_blend_train.csv',index=False)
blend_test.to_csv('./data/dp_blend_test.csv',index=False)

# save y_pred 
pcc = list(cv_score['pearsonr_score'])
s,y_pred = sum(pcc),0
for i in range(len(pcc)):
    y_pred += cv_y_pred[i]* (pcc[i]/s)

res = pd.DataFrame()
res['id'],res['target'] = test_id,y_pred
from datetime import datetime
datetime = datetime.now()
output_path = './data/dp_%.4f_%d.csv'%(cv_score.mean()['pearsonr_score'],datetime.microsecond)
res.to_csv(output_path,index=False,sep=' ',header=False)


## using blend to with last weeek old blend_train

blend_trian,blend_test = None,None
blend_train_cur,blend_test_cur = pd.read_csv('./data/blend_train.csv'),pd.read_csv("./data/blend_test.csv")
blend_train,blend_test = blend_train_cur,blend_test_cur
blend_train_cur,blend_test_cur = pd.read_csv('./data/dp_blend_train.csv'),pd.read_csv("./data/dp_blend_test.csv")
blend_train,blend_test = pd.merge(blend_train,blend_train_cur,on='id',how='left',copy=False),pd.merge(blend_test,blend_test_cur,on='id',how='left',copy=False)
target  = pd.read_csv('./data/train_target.csv',sep='\t',header=None,names=['id','target'])
blend_train = pd.merge(blend_train,target,on='id',how='left',copy=False)
blend_test['target'] = -5
data = pd.concat([blend_train,blend_test],copy=False)


t0 = time()
print('Ridge train...')

train,test = data[data['target']!=-5],data[data['target']==-5]

train_id, y= train.pop('id'),train.pop('target')
col = train.columns
X = train[col].values

test_id,test_y= test.pop('id'),test.pop('target')
test = test[col].values

N = 10
skf = KFold(n_splits=N,shuffle=False,random_state=2333)

xx_cv,xx_pre = [],[]

for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
    reg = Ridge(alpha=.1)
    reg.fit(X_train,y_train)
    print('Start predicting...')
    y_pred = reg.predict(X_test)
    evs,mae,mse,r2 = explained_variance_score(y_test,y_pred),mean_absolute_error(y_test,y_pred),mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)
    pcc = stats.pearsonr(y_test,y_pred)[0]
    xx_cv.append([evs,mae,mse,r2,pcc])
    print('pearsonr:',pcc)
    xx_pre.append(reg.predict(test))
    
print('**'*20,'blend',' 10-Fold CV:','**'*20)
xx_cv = np.mean(xx_cv,axis=0).reshape(1,5)
xx_cv = pd.DataFrame(xx_cv,columns=['explained_variance_score','mae','mse','r2_score','pearsonr_score'])
print(xx_cv)

s = 0
for i in xx_pre: s = s + i
s = s /N

res = pd.DataFrame()
res['id'] = list(test_id.values)
res['RST'] = list(s)

print('xx_cv',np.mean(xx_cv))
print('Ridge train done in %.3fs'%(time()-t0))
datetime = datetime.now()
res.to_csv('./data/ret_blend_ridge%d_%s.csv'%(datetime.microsecond,str(xx_cv['pearsonr_score'][0]).split('.')[1]),index=False,sep=' ',header=False)

