import os
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from scipy import stats
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import Ridge
pd.options.display.max_columns = 999


def opensmile_extract(audio_path,out_path,config,opensmile_path='D:/Program/opensmile'):
    '''using opensmile to extract feature from audio file(.wav)
    Parameters
    audio_path: String, the directory path of audios
    ouput_path: String, the dirctory path of ouput files
    config: String the cofig name
    Returns
    '''
    t0 = time()
    print('opensmile extract...')
    audio_list=os.listdir(audio_path)
    for audio in audio_list:
        if audio[-4:]=='.wav':
            this_path_input=os.path.join(audio_path,audio)
            this_path_output=os.path.join(output_path,audio[:-4]+'.txt')
            cmd='cd /d '+opensmile_path+'/bin/Win32 && SMILExtract_Release -C\
             D:/Program/opensmile/config/'+config+'.conf -I '+this_path_input+' -O '+this_path_output
        os.system(cmd)
    print('opensmile extract done in %0.3fs'%(time()-t0))

def get_feature(opensmile_result_path):
    ''' get feature from files which created by opensmile
    Parameters: 
    opensmile_result_path: String, directory path of file which created by opensmile
    Return:
    features_df: pandas.DataFrame, the faetures from directory files.
    '''
    t0 = time()
    print('get features from opensmile result....')
    txt_path=opensmile_result_path
    txt_list=os.listdir(txt_path)
    features_names = []
    with open(os.path.join(txt_path,txt_list[0])) as f:
        for line in f.readlines():
            if '@attribute' in line: features_names.append(line[11:])
    features_names = ['id']+features_names[1:-1]
    features_list=[]
    for txt in txt_list:
        if txt[-4:]=='.txt':
            this_path=os.path.join(txt_path,txt)
            f=open(this_path)
            last_line=f.readlines()[-1]
            f.close()
            features = [txt[:-4]]
            features +=last_line.split(',')[1:-1]
            features_list.append(features)
    features_df = pd.DataFrame(features_list,columns=features_names)
    print('get features done in %.3fs'%(time()-t0))
    return features_df
def pearsonr_score(preds,train_data):
    ''' Customized evaluation function to using lightgbm.train
    Parameters:
    preds: 1d array-like, the predict of model
    train_data: lightgbm.Dataset
    Returns:
    (eval_name, eval_result, is_higher_better)
    '''
    labels = train_data.get_label()
    return 'pearsonr',stats.pearsonr(preds,labels)[0],True

if not os.path.exists('./data'):
    os.makedirs('./data')
configs = ['IS09_emotion','IS10_paraling','IS11_speaker_state','IS12_speaker_trait','IS13_ComParE', 'emobase','emobase2010','emo_large']
#configs = ['emobase2010','emo_large']
blend_train,blend_test = [],[]
train_id,test_id = None,None
score_df = []

for config in configs:
    print('=='*20,config,'=='*20)
    for path in ['train','test']:
        audio_path='F:/work/internship/audio_data/%s/audio'%(path)
        output_path='F:/work/internship/audio_data/%s/%s'%(path,config)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        opensmile_extract(audio_path,output_path,config)
        features_df = get_feature(output_path)
        features_df.to_csv('./data/%s_%s.csv'%(config,path),index=False)


    train_data,test_data = pd.read_csv('./data/%s_train.csv'%(config)),pd.read_csv('./data/%s_test.csv'%(config))
    train_target = pd.read_csv('./data/train_target.csv',sep='\t',header=None,names=['id','target'])
    test_data['target'] = -5
    train_data = pd.merge(train_data,train_target,on='id',how='left',copy=False)
    data = pd.concat([train_data,test_data],copy=False)
    data = data.sort_values(by='id')

    t0 = time()
    print('lgbm train...')
    train,test = data[data['target']!=-5],data[data['target']==-5]

    blend_train.append(np.zeros((len(train),1)) )
    blend_test.append(np.zeros((len(test),1)) )

    train_id, y= train.pop('id'),train.pop('target')
    col = train.columns
    X = train[col].values

    test_id,test_y= test.pop('id'),test.pop('target')
    test = test[col].values

    N = 10
    skf = KFold(n_splits=N,shuffle=False,random_state=2333)

    xx_cv = []

    for train_in,test_in in skf.split(X,y):
        X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 32,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=4000,
                        valid_sets=lgb_eval,
                        feval = pearsonr_score,
                        verbose_eval=100,
                        early_stopping_rounds=50)

        print('Start predicting...')
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        blend_train[-1][test_in] = np.asarray(y_pred).reshape((-1,1))
        
        evs,mae,mse,r2 = explained_variance_score(y_test,y_pred),mean_absolute_error(y_test,y_pred),mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)
        pcc = stats.pearsonr(y_test,y_pred)[0]
        xx_cv.append([evs,mae,mse,r2,pcc])
        y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
        blend_test[-1] =blend_test[-1]+ np.asarray(y_pred).reshape((-1,1))/N

    print('lightgbm train done in %.3fs'%(time()-t0))
    print('**'*20,config,' 5-Fold:','**'*20)
    xx_cv = np.mean(xx_cv,axis=0).reshape(1,5)
    score_df.append(xx_cv)
    xx_cv = pd.DataFrame(xx_cv,columns=['explained_variance_score','mae','mse','r2_score','pearsonr_score'])
    print(xx_cv)

    res = pd.DataFrame()
    res['id'] = list(test_id.values)
    res['RST'] = blend_test[-1]
    datetime = datetime.now()
    res.to_csv('./data/%s_%d_%s.csv'%(config,datetime.microsecond,str(xx_cv['pearsonr_score'][0]).split('.')[1]),index=False,sep=' ',header=False)

backup_train,backup_tes = blend_train,blend_test
blend_train,blend_test = np.concatenate(blend_train,axis=1),np.concatenate(blend_test,axis=1)
blend_train,blend_test = pd.DataFrame(blend_train),pd.DataFrame(blend_test)
blend_train['id'],blend_test['id'] = list(train_id),list(test_id)
blend_train.to_csv('./data/blend_train.csv',index=False)
blend_test.to_csv('./data/blend_test.csv',index=False)

#blend_train,blend_test = pd.read_csv('./data/blend_train.csv'),pd.read_csv('./data/blend_test.csv')
target  = pd.read_csv('./data/train_target.csv',sep='\t',header=None,names=['id','target'])
blend_train = pd.merge(blend_train,target,on='id',how='left',copy=False)
blend_test['target'] = -5
data = pd.concat([blend_train,blend_test],copy=False)

''' to see all metrics from above predict '''
# cols = [s for s in blend_train if s!='id' and s!='target']
# configs = ['IS09_emotion','IS10_paraling','IS11_speaker_state','IS12_speaker_trait','IS13_ComParE', 'emobase','emobase2010','emo_large']
# y_test = target['target'].values
# for i in range(len(cols)):
#     xx_cv = []
#     col,config = cols[i],configs[i]
#     y_pred = blend_train[col].values
#     evs,mae,mse,r2 = explained_variance_score(y_test,y_pred),mean_absolute_error(y_test,y_pred),mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)
#     pcc = stats.pearsonr(y_test,y_pred)[0]
#     xx_cv.append([evs,mae,mse,r2,pcc])
#     print('**'*20,config,' 5-Fold CV:','**'*20)
#     xx_cv = np.mean(xx_cv,axis=0).reshape(1,5)
#     score_df.append(xx_cv)
#     xx_cv = pd.DataFrame(xx_cv,columns=['explained_variance_score','mae','mse','r2_score','pearsonr_score'])
#     print(xx_cv)

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
    
print('**'*20,'blend',' 5-Fold CV:','**'*20)
xx_cv = np.mean(xx_cv,axis=0).reshape(1,5)
score_df.append(xx_cv)
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
score_df = np.concatenate(score_df,axis=0)
score_df = pd.DataFrame(score_df,columns=['explained_variance_score','mae','mse','r2_score','pearsonr_score'])
cols = score_df.columns.tolist()
score_df['method'] = ['IS09_emotion-lightgbm','IS10_paraling-lightgbm','IS11_speaker_state-lightgbm','IS12_speaker_trait-lightgbm','IS13_ComParE-lightgbm', 'emobase-lightgbm','emobase2010-lightgbm','emo_large-lightgbm','blend-ridge']
cols = ['method'] + cols
score_df = score_df[cols]
score_df.to_csv('./data/score_df.csv')