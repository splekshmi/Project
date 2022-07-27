import pandas as pd
import pickle
import xgboost as xgb
df=pd.read_csv('file5.csv')
df.sort_values(["ageEstimate"], 
                    axis=0,
                    ascending=[True], 
                    inplace=True)
X=df.drop(['avgCompanyPosDuration','Unnamed: 0'],axis=1)
y=df['avgCompanyPosDuration']
from sklearn import preprocessing
label_encoder1 = preprocessing.LabelEncoder().fit(X['genderEstimate'].astype(str))
X['genderEstimate']= label_encoder1.transform(X['genderEstimate'].astype(str))
label_encoder2 = preprocessing.LabelEncoder().fit(X['mbrLocation'].astype(str))
X['mbrLocation']= label_encoder2.transform(X['mbrLocation'].astype(str))
label_encoder3 = preprocessing.LabelEncoder().fit(X['companyName'].astype(str))
X['companyName']= label_encoder3.transform(X['companyName'].astype(str))
label_encoder4 = preprocessing.LabelEncoder().fit(X['posTitle'].astype(str))
X['posTitle']= label_encoder4.transform(X['posTitle'].astype(str))
label_encoder5 = preprocessing.LabelEncoder().fit(X['posLocation'].astype(str))
X['posLocation']= label_encoder5.transform(X['posLocation'].astype(str))
pickle.dump(label_encoder1,open('l1.pkl','wb') )
pickle.dump(label_encoder2,open('l2.pkl','wb') )
pickle.dump(label_encoder3,open('l3.pkl','wb') )
pickle.dump(label_encoder4,open('l4.pkl','wb') )
pickle.dump(label_encoder5,open('l5.pkl','wb') )
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20)
r =xgb.XGBRegressor(random_state = 42,n_estimators = 1000,n_jobs=-1,max_leaves= 5,learning_rate=0.1,min_child_weight=1,max_depth=10,gamma=0.02,subsample=1,reg_alpha=0,scale_pos_weight= 1)
m=r.fit(X_train,y_train)
pickle.dump(r,open('model.pkl','wb') )