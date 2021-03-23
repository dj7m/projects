from flask import Flask,render_template,request
import joblib
import os
import numpy as np
import pickle

app=Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/result",methods=['POST','GET'])
def result():
    age=float(request.form['age'])
    ejection_fraction=int(request.form['ejection_fraction'])
    serum_creatinine=float(request.form['serum_creatinine'])
    serum_sodium=int(request.form['serum_sodium'])
    time=int(request.form['time'])
    x=np.array([age,ejection_fraction,serum_creatinine,serum_sodium,time]).reshape(1,-1)


    model_path=os.path.join('D:\\projects\\stroke prediction','models\\rfc.sav')
    rfc=joblib.load(model_path)

    p3=rfc.predict(x)
    if p3==0:
        render_template('nontroke.html')
    else:
        return render_template('stroke.html')

if __name__=="__main__":
   app.run(debug=True,port=7384)