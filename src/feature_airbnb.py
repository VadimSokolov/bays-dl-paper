#!/usr/bin/env python 

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

print('Loading raw data...')
train_users_path = '../data/airbnb/train_users.csv'
test_users_path = '../data/airbnb/test_users.csv'
sessions_path = '../data/airbnb/sessions.csv'

# Note: age_gender_bkts.csv and countries.csv files are not used.

# ######## Loading data #############
# train_users
df_train = pd.read_csv(train_users_path)
target = df_train['country_destination']
df_train = df_train.drop(['country_destination'], axis=1)

# test_users
df_test = pd.read_csv(test_users_path)
id_test = df_test['id']

# sessions
df_sessions = pd.read_csv(sessions_path)
df_sessions['id'] = df_sessions['user_id']
df_sessions = df_sessions.drop(['user_id'], axis=1)

#########Preparing Session data########
print('Working on Session data...')
#Filling nan with specific value ('NAN')
df_sessions.action = df_sessions.action.fillna('NAN')
df_sessions.action_type = df_sessions.action_type.fillna('NAN')
df_sessions.action_detail = df_sessions.action_detail.fillna('NAN')
df_sessions.device_type = df_sessions.device_type.fillna('NAN')

#Action values with low frequency are changed to 'OTHER'
act_freq = 100  #Threshold for frequency
act = dict(zip(*np.unique(df_sessions.action, return_counts=True)))
df_sessions.action = df_sessions.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x)

#Computing value_counts. These are going to be used in the one-hot encoding
#based feature generation (following loop).
f_act = df_sessions.action.value_counts().argsort()
f_act_detail = df_sessions.action_detail.value_counts().argsort()
f_act_type = df_sessions.action_type.value_counts().argsort()
f_dev_type = df_sessions.device_type.value_counts().argsort()


#grouping session by id. We will compute features from all rows with the same id.
dgr_sess = df_sessions.groupby(['id'])

#Loop on dgr_sess to create all the features.
samples = []
cont = 0
ln = len(dgr_sess)
for g in dgr_sess:
    if cont%10000 == 0:
        print("%s from %s" %(cont, ln))
    gr = g[1]
    l = []
    
    #the id
    l.append(g[0])
    
    #The actual first feature is the number of values.
    l.append(len(gr))
    
    sev = gr.secs_elapsed.fillna(0).values   #These values are used later.
    
    #action features
    #(how many times each value occurs, numb of unique values, mean and std)
    c_act = [0] * len(f_act)
    for i,v in enumerate(gr.action.values):
        c_act[f_act[v]] += 1
    _, c_act_uqc = np.unique(gr.action.values, return_counts=True)
    c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]
    l = l + c_act
    
    #action_detail features
    #(how many times each value occurs, numb of unique values, mean and std)
    c_act_detail = [0] * len(f_act_detail)
    for i,v in enumerate(gr.action_detail.values):
        c_act_detail[f_act_detail[v]] += 1 
    _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)
    c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]
    l = l + c_act_detail
    
    #action_type features
    #(how many times each value occurs, numb of unique values, mean and std
    #+ log of the sum of secs_elapsed for each value)
    l_act_type = [0] * len(f_act_type)
    c_act_type = [0] * len(f_act_type)
    for i,v in enumerate(gr.action_type.values):
        l_act_type[f_act_type[v]] += sev[i]   
        c_act_type[f_act_type[v]] += 1  
    l_act_type = np.log(1 + np.array(l_act_type)).tolist()
    _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)
    c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]
    l = l + c_act_type + l_act_type    
    
    #device_type features
    #(how many times each value occurs, numb of unique values, mean and std)
    c_dev_type  = [0] * len(f_dev_type)
    for i,v in enumerate(gr.device_type .values):
        c_dev_type[f_dev_type[v]] += 1 
    c_dev_type.append(len(np.unique(gr.device_type.values)))
    _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)
    c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]        
    l = l + c_dev_type    
    
    #secs_elapsed features        
    l_secs = [0] * 5 
    l_log = [0] * 15
    if len(sev) > 0:
        #Simple statistics about the secs_elapsed values.
        l_secs[0] = np.log(1 + np.sum(sev))
        l_secs[1] = np.log(1 + np.mean(sev)) 
        l_secs[2] = np.log(1 + np.std(sev))
        l_secs[3] = np.log(1 + np.median(sev))
        l_secs[4] = l_secs[0] / float(l[1])
        
        #Values are grouped in 15 intervals. Compute the number of values
        #in each interval.
        log_sev = np.log(1 + sev).astype(int)
        l_log = np.bincount(log_sev, minlength=15).tolist()                      
    l = l + l_secs + l_log
    
    #The list l has the feature values of one sample.
    samples.append(l)
    cont += 1

#Creating a dataframe with the computed features    
col_names = []    #name of the columns
for i in range(len(samples[0])-1):
    col_names.append('c_' + str(i)) 
#preparing objects    
samples = np.array(samples)
samp_ar = samples[:, 1:].astype(np.float16)
samp_id = samples[:, 0]   #The first element in obs is the id of the sample.

#creating the dataframe        
df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)
df_agg_sess['id'] = samp_id
df_agg_sess.index = df_agg_sess.id

#########Working on train and test data#####################
print('Working on users data...')
#Concatenating df_train and df_test
df_tt = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_tt.index = df_tt.id
df_tt = df_tt.fillna(-1)  #Inputing this kind of missing value with -1 (missing values in train and test)
df_tt = df_tt.replace('-unknown-', -1) #-unknown is another way of missing value, then = -1.

########Creating features for train+test
#Removing date_first_booking
df_tt = df_tt.drop(['date_first_booking'], axis=1)

#Number of nulls
df_tt['n_null'] = np.array([sum(r == -1) for r in df_tt.values])

#date_account_created
#(Computing year, month, day, week_number, weekday)
dac = np.vstack(df_tt.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_tt['dac_y'] = dac[:,0]
df_tt['dac_m'] = dac[:,1]
df_tt['dac_d'] = dac[:,2]
dac_dates = [datetime(x[0],x[1],x[2]) for x in dac]
df_tt['dac_wn'] = np.array([d.isocalendar()[1] for d in dac_dates])
df_tt['dac_w'] = np.array([d.weekday() for d in dac_dates])
df_tt_wd = pd.get_dummies(df_tt.dac_w, prefix='dac_w')
df_tt = df_tt.drop(['date_account_created', 'dac_w'], axis=1)
df_tt = pd.concat((df_tt, df_tt_wd), axis=1)

#timestamp_first_active
#(Computing year, month, day, hour, week_number, weekday)
tfa = np.vstack(df_tt.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_tt['tfa_y'] = tfa[:,0]
df_tt['tfa_m'] = tfa[:,1]
df_tt['tfa_d'] = tfa[:,2]
df_tt['tfa_h'] = tfa[:,3]
tfa_dates = [datetime(x[0],x[1],x[2],x[3],x[4],x[5]) for x in tfa]
df_tt['tfa_wn'] = np.array([d.isocalendar()[1] for d in tfa_dates])
df_tt['tfa_w'] = np.array([d.weekday() for d in tfa_dates])
df_tt_wd = pd.get_dummies(df_tt.tfa_w, prefix='tfa_w')
df_tt = df_tt.drop(['timestamp_first_active', 'tfa_w'], axis=1)
df_tt = pd.concat((df_tt, df_tt_wd), axis=1)

#timespans between dates
#(Computing absolute number of seconds of difference between dates, sign of the difference)
df_tt['dac_tfa_secs'] = np.array([np.log(1+abs((dac_dates[i]-tfa_dates[i]).total_seconds())) for i in range(len(dac_dates))])
df_tt['sig_dac_tfa'] = np.array([np.sign((dac_dates[i]-tfa_dates[i]).total_seconds()) for i in range(len(dac_dates))])
#    df_tt['dac_tfa_days'] = np.array([np.sign((dac_dates[i]-tfa_dates[i]).days) for i in range(len(dac_dates))])

#Comptute seasons from dates
#(Computing the season for the two dates)
Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [(0, (date(Y,  1,  1),  date(Y,  3, 20))),  #'winter'
           (1, (date(Y,  3, 21),  date(Y,  6, 20))),  #'spring'
           (2, (date(Y,  6, 21),  date(Y,  9, 22))),  #'summer'
           (3, (date(Y,  9, 23),  date(Y, 12, 20))),  #'autumn'
           (0, (date(Y, 12, 21),  date(Y, 12, 31)))]  #'winter'
def get_season(dt):
    dt = dt.date()
    dt = dt.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= dt <= end)
df_tt['season_dac'] = np.array([get_season(dt) for dt in dac_dates])
df_tt['season_tfa'] = np.array([get_season(dt) for dt in tfa_dates])
#df_all['season_dfb'] = np.array([get_season(dt) for dt in dfb_dates])

#Age
#(Keeping ages in 14 < age < 99 as OK and grouping others according different kinds of mistakes)
av = df_tt.age.values
av = np.where(np.logical_and(av<2000, av>1900), 2014-av, av) #This are birthdays instead of age (estimating age by doing 2014 - value)
av = np.where(np.logical_and(av<14, av>0), 4, av) #Using specific value=4 for age values below 14
av = np.where(np.logical_and(av<2016, av>2010), 9, av) #This is the current year insted of age (using specific value = 9)
av = np.where(av > 99, 110, av)  #Using specific value=110 for age values above 99
df_tt['age'] = av

#AgeRange
#(One-hot encoding of the edge according these intervals)
interv =  [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
def get_interv_value(age):
    iv = 20
    for i in range(len(interv)):
        if age < interv[i]:
            iv = i 
            break
    return iv
df_tt['age_interv'] = df_tt.age.apply(lambda x: get_interv_value(x))
df_tt_ai = pd.get_dummies(df_tt.age_interv, prefix='age_interv')
df_tt = df_tt.drop(['age_interv'], axis=1)
df_tt = pd.concat((df_tt, df_tt_ai), axis=1)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_tt_dummy = pd.get_dummies(df_tt[f], prefix=f)
    df_tt = df_tt.drop([f], axis=1)
    df_tt = pd.concat((df_tt, df_tt_dummy), axis=1)    
   
######Merging train-test with session data#################
df_all = pd.merge(df_tt, df_agg_sess, how='left')
df_all = df_all.drop(['id'], axis=1)
df_all = df_all.fillna(-2)  #Missing features for samples without sesssion data.
#All types of null 
df_all['all_null'] = np.array([sum(r<0) for r in df_all.values])


######Computing X, y and X_test ################
piv_train = len(target) #Marker to split df_all into train + test
vals = df_all.values
le = LabelEncoder()

X = vals[:piv_train]
y = le.fit_transform(target.values)
X_test = vals[piv_train:]
print('Shape X = %s, Shape X_test = %s'%(X.shape, X_test.shape))

with open("../data/airbnb/X.pickle", 'wb') as fh:
# with open("/Users/vsokolov/Downloads/X5k.pickle", 'wb') as fh:
	pickle.dump(X, fh)
with open("../data/airbnb/X_test.pickle", 'wb') as fh:
	pickle.dump(X_test, fh)
with open("../data/airbnb/y.pickle", 'wb') as fh:
# with open("/Users/vsokolov/Downloads/Y5k.pickle", 'wb') as fh:
	pickle.dump(y, fh)


# with open("../data/airbnb/X.pickle", 'rb') as fh:
# 	X = pickle.load(fh)
# with open("../data/airbnb/X_test.pickle", 'rb') as fh:
# 	X_test = pickle.load(fh)
# with open("../data/airbnb/y.pickle", 'rb') as fh:
# 	y = pickle.load(fh)
