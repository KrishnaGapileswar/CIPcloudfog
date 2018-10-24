from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS, cross_origin
import os
import requests
import time
import irismodel
import json
from sklearn.externals import joblib
import csv
import matplotlib

matplotlib.use('agg', warn=False, force=True)
from matplotlib import pyplot as plt

numSamples = 0

SVM_FOLDER = os.path.join('static','svm_photo')

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = SVM_FOLDER

@app.route('/train',methods=['POST'])
@cross_origin()
def train():
        Cparam = request.json["Cparam"]
        accuracy = irismodel.train(Cparam)
        global numSamples
        numSamples = accuracy[1]
        print(numSamples)
        return ("Training Success and Accuracy is " + str(accuracy[0]['accuracy']))

@app.route('/showgraph',methods=['GET'])
@cross_origin()
def showgraph():
        counter = 1
        x=[]
        y=[]
        with open('time.csv','r') as t:
                plots = csv.reader(t, delimiter=',')
                for row in plots:
                        x.append(counter)
                        y.append(float(row[0]))
                        counter+=1
        plt.plot(x,y, label='Time plot')
        plt.xlabel('Iterations')
        plt.ylabel('Time in milliseconds')
        plt.savefig('./static/svm_photo/timegraph.png')
        imgpath = os.path.join(app.config['UPLOAD_FOLDER'], 'timegraph.png')
        imgpath = 'http://192.168.43.210/'+imgpath
        return (imgpath)

@app.route('/showtraingraph',methods=['GET'])
@cross_origin()
def showtraingraph():
        x=[]
        y=[]
        with open('traintime.csv','r') as t:
                plots = csv.reader(t, delimiter=',')
                for row in plots:
                        x.append(row[0])
                        y.append(float(row[1]))
        plt.plot(x,y, label='Time plot')
        plt.xlabel('No. of samples')
        plt.ylabel('Time in milliseconds')
        plt.savefig('./static/svm_photo/traintimegraph.png')
        imgpath = os.path.join(app.config['UPLOAD_FOLDER'], 'traintimegraph.png')
        imgpath = 'http://192.168.43.210/'+imgpath
        return (imgpath)

@app.route('/savetime',methods=['POST'])
@cross_origin()
def savetime():
        delay = float(request.json["delay"])
        with open('./time.csv','a') as timefile:
                twriter = csv.writer(timefile, delimiter=',')
                twriter.writerow([delay])
        return "1"


@app.route('/savetraintime',methods=['POST'])
@cross_origin()
def savetraintime():
        delay = float(request.json["delay"])
        with open('./traintime.csv','a') as traintimefile:
                twriter = csv.writer(traintimefile, delimiter=',')
                print(numSamples)
                twriter.writerow([int(numSamples), delay])
        return "1"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
        sl = float(request.json["sepalLength"])
        sw = float(request.json["sepalWidth"])
        pl = float(request.json["petalLength"])
        pw = float(request.json["petalWidth"])
        X = [[sl, sw, pl, pw]]
        clf = joblib.load('model.pkl')
        probabilities = clf.predict_proba(X)
        data = [{'name': 'Iris-Setosa', 'value': round(probabilities[0, 0] * 100, 2)},
            {'name': 'Iris-Versicolour', 'value': round(probabilities[0, 1] * 100, 2)},
            {'name': 'Iris-Virginica', 'value': round(probabilities[0, 2] * 100, 2)}]
        return jsonify(data)

@app.route('/')
@app.route('/index')
@cross_origin()
def index():
        return render_template('index.html')

if __name__ == '__main__':
        app.run(host = '0.0.0.0',debug=True,port = 8081)