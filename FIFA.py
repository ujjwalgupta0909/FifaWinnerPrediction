# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 00:15:38 2019

@author: hp
"""
import pandas as pd
import numpy as np
wc=pd.read_csv('World_Cup_2018_Dataset.csv') 
features1=wc.iloc[0:32,0:19].values
features1=pd.DataFrame(features1)

"""working on RESULTS dataset"""
results=pd.read_csv('results.csv')
results=results.iloc[:,0:5].values
results=pd.DataFrame(results,columns=['date','home_team','away_team','home_score','away_score'])
Impfeatures2=pd.DataFrame(columns=['date','home_team','away_team','home_score','away_score'])
random_matches=pd.DataFrame(columns=['date','home_team','away_team','home_score','away_score'])
Curr=pd.DataFrame(columns=['date','home_team','away_team','home_score','away_score'])


#Impfeatures2=Impfeatures2.append(features2.iloc[[0],:])

for name in features1[0]:
    n=name
    for i in range(0,39070):
        if n in results.iloc[i,1]:
            Impfeatures2=Impfeatures2.append(results.iloc[[i],:])

Impfeatures2=Impfeatures2[Impfeatures2.home_team != 'Korea DPR']
Impfeatures2=Impfeatures2[Impfeatures2.away_team != 'Korea DPR']
#Impfeatures2 =Impfeatures2.drop(Impfeatures2.home_team="Korea DPR")          
Impfeatures2.index= range(10834)

            
'''for name in features1[0]:
    n=name
    for i in range(0,19738):
        if n not in Impfeatures2.iloc[i,1]:
            Impfeatures2=Impfeatures2.drop(Impfeatures2.iloc[[i],:])'''

for name in features1[0]:
    n=name
    for i in range(0,10834):
        if n in Impfeatures2.iloc[i,2]:
            random_matches=random_matches.append(Impfeatures2.iloc[[i],:])
            
random_matches.index= range(3983)
#if winner is awayteam then 2
#if winner is hometeam then 1
#if draw then 0
match_result=[]
for j in range(0,3983):
    if (random_matches['home_score'][j]<random_matches['away_score'][j]):
        match_result.append(2)
    if (random_matches['home_score'][j]>random_matches['away_score'][j]):
        match_result.append(1)
    elif (random_matches['home_score'][j]==random_matches['away_score'][j]):
        match_result.append(0)

random_matches['winning_team'] = match_result
random_matches=random_matches.iloc[:,[1,2,5]]
qq=random_matches
random_matches=qq
#X=random_matches.iloc[:,:-1]
X0=random_matches.iloc[:,[0]]
X1=random_matches.iloc[:,[1]]
#taking care of categorical data

#final = pd.get_dummies(random_matches, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
#final = pd.get_dummies(random_matches, prefix=['home_team'], columns=['home_team'])
#finall = pd.get_dummies(random_matches, prefix=['away_team'], columns=['away_team'])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X0=X0.apply(labelencoder.fit_transform)
X1=X1.apply(labelencoder.fit_transform)
#random_matches.values[:,0]=labelencoder.fit_transform(random_matches.values[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X0=onehotencoder.fit_transform(X0).toarray()
X0=X0[:,1:]
onehotencoderr=OneHotEncoder(categorical_features=[0])
X1=onehotencoderr.fit_transform(X1).toarray()
X1=X1[:,1:]

X0=pd.DataFrame(X0)
X1=pd.DataFrame(X1)
X=pd.concat([X0,X1],axis=1)
#X=X[:,:-1]

Y=random_matches.iloc[:,[2]]

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X,Y)

def predictnow(hometeam,awayteam):
    ax=labelencoder.transform(hometeam)  #hometeam
    bx=labelencoder.transform(awayteam)   #awayteam
    ohe=onehotencoder.transform(ax).toarray()
    ohe=pd.DataFrame(ohe)
    ohe=ohe.iloc[:,1:]
    ohee=pd.DataFrame(onehotencoderr.transform(bx).toarray())
    ohee=ohee.iloc[:,1:]
    oh=pd.concat([ohe,ohee],axis=1)
    print(classifier.predict(oh))
    
"""examples"""
predictnow('Russia','Saudi Arabia')
predictnow('Egypt','Uruguay')
predictnow('Morocco','Iran')
predictnow('Portugal','Spain')
predictnow('France','Australia')
predictnow('Argentina','Iceland')
predictnow('Peru','Denmark')
