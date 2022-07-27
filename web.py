from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
   co= request.values['county']
   st= request.values['state']
   cy= request.values['city']
   yr= request.values['year']
   mapy={1:'poor',2:'Moderate',3:'Good'}
   query=np.array([co,st,cy,yr])
   query=query.astype(np.float64)
   query=query.reshape(1,-1)
   out=mapy[model.predict(query)[0]]
   return render_template ('result.html',prediction_text="The Quality of Air is." ,name=out)
   #output=model.predict(exp1)
   #return render_template ('result.html',)
if __name__=='__main__':
    app.run()

