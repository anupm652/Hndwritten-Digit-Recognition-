#Load all the necessary libraries
from flask import Flask, render_template, request
import joblib
import numpy as np
import cv2

#Load the saved model
ml_model=joblib.load("hrdr.pkl")

#Start the flask application
app=Flask(__name__)

#defined some functions
def predict_label(img_path):
    i=cv2.imread(img_path,0)
    i=np.invert(np.array([i]))
    i=i.reshape(1,784)
    p=ml_model.predict(i)
    return p[0]

#Define routes for specific tasks
@app.route('/',methods=["GET","POST"])
def home():
    return render_template("index.html")

@app.route("/submit",methods=["GET","POST"])
def get_output():
    if request.method =="POST":
        img=request.files["Image"]
        img_path ="static/"+img.filename
        img.save(img_path)
        p =predict_label(img_path)
    return render_template("index.html",prediction=p,img_path=img_path)

# Standard thing xD
if __name__=="__main__":
    app.run(debug=True)
