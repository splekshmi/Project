# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:30:46 2022

@author: param
"""

from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
l1=pickle.load(open('l1.pkl','rb'))
l2=pickle.load(open('l2.pkl','rb'))
l3=pickle.load(open('l3.pkl','rb'))
l4=pickle.load(open('l4.pkl','rb'))
l5=pickle.load(open('l5.pkl','rb'))
df=pd.read_csv('file5.csv')
df.sort_values(["ageEstimate"], 
                    axis=0,
                    ascending=[True], 
                    inplace=True)
@app.route('/',methods=['GET'])
def home():
    return render_template('home.html',data=df)
@app.route('/predict',methods=['GET','POST'])
def predict(): 
    t_Age = request.form.get('t_Age')
    t_Gender = request.form.get('t_Gender')
    t_mbrLocation = request.form.get('t_mbrLocation')
    t_Company = request.form.get('t_Company')
    t_companyStaffCount = request.form.get('t_companyStaffCount')
    t_posTitle = request.form.get('t_posTitle')
    t_posLocation = request.form.get('t_posLocation')
   
  
    test=pd.DataFrame([[ t_Age,t_Gender,t_mbrLocation,t_Company,t_companyStaffCount,t_posTitle,t_posLocation]],columns=['ageEstimate','genderEstimate','mbrLocation','companyName','companyStaffCount','posTitle','posLocation'])
    
    test['ageEstimate']=float(test['ageEstimate'])
    test['companyStaffCount']=float(test['companyStaffCount'])
    
    test['genderEstimate']=l1.transform(test['genderEstimate'])
    test['mbrLocation']=l2.transform(test['mbrLocation'])
    test['companyName']=l3.transform(test['companyName'])
    test['posTitle']=l4.transform(test['posTitle'])
    test['posLocation']=l5.transform(test['posLocation'])
    

    #return render_template ('result.html',prediction_text="The encoded values are post location is {}".format(test))
    
    prediction =model.predict(test)
    prediction=prediction.item()
    prediction=np.round(prediction,2)
    
    return render_template ('result.html',prediction_text="The employee tenure duration is {} years".format(prediction))
if __name__=='__main__':
  app.run(port=8000)