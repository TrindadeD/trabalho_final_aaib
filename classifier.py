import time
import json
import socket
import numpy as np
import pandas as pd
import tsfel
from joblib import load

# from sklearn.esemble import RandomForestClassifier
import sklearn

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Thread, Event

app = Flask(__name__)
socketio = SocketIO(app)

thread = Thread()
thread_stop_event = Event()


def event_stream():

    

    def get_magnitude(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        host = "192.168.1.163" #ip of smartphone
        port = 4242 #defined on app on phone
        s.connect((host, port))


        # data acquisition
        # returns file new_data.csv
        filename="new_data"
        sample=100
        
    
        f=open(filename+'.csv', 'w')
        for n in range(sample):
            #time.sleep(0.025)
            data = s.recv(256)
            if data:
                decoded_data = data.decode("utf-8").split("\n")
                for msg in decoded_data:
                    try:
                        package = json.loads(msg)
                        #print(package)
                        t=package["accelerometer"]["timestamp"]
                        acc=package["accelerometer"]["value"]
                        gyro=package["gyroscope"]["value"]
                        socketio.emit('serverResponse', {'timestamp': time.time(), 'data': get_magnitude(package["accelerometer"]["value"])})
                        a = str(t)+','+str(acc[0])+','+str(acc[1])+','+str(acc[2])+','+str(gyro[0])+','+str(gyro[1])+','+str(gyro[2])
                        print(a)
                        f.write(a+'\n')
    
    
                    except:
                        continue
    
    
        f.close()


# passar valores iniciais ???
@app.route('/')
def sessions():
    return render_template('index.html')
    

# File processing
# Feature extraction
# Feature selection
# Classification
# returns the class -> activity 
def process(filename):
    # configuration file -> file with selected features
    cfg_file = tsfel.load_json('features.json') # same directory as app.py
    # save in a data frame
    df = pd.read_csv("new_data.csv",  names=['timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'])
    df = df.drop(['timestamp'], 1)
    df_features = tsfel.time_series_features_extractor(cfg_file, df)
    #feature selection
    corr_feat =['0_Autocorrelation', '0_Standard deviation', '0_Variance', '1_Autocorrelation', '1_Peak to peak distance', '1_Standard deviation', '1_Variance', '2_Autocorrelation', '2_Peak to peak distance', '2_Standard deviation', '2_Variance', '3_Autocorrelation', '3_Peak to peak distance', '3_Standard deviation', '3_Variance', '4_Autocorrelation', '4_Peak to peak distance', '4_Standard deviation', '4_Variance', '5_Autocorrelation', '5_Peak to peak distance', '5_Standard deviation', '5_Variance']
    df_features.drop(corr_feat, axis=1, inplace=True)
    df_features = np.array(df_features)
    classifier = load('randomforest.json') # same directory as app.py
    # Predict -> activity being performed
    prediction = classifier.predict(df_features) # 1, 2, 3, 4 
    # case
    if prediction == "1":
        code = "dentro_pe"
    if prediction == "2":
        code = "fora_pe"
    if prediction == "3":
        code = "pentear"
    if prediction == "4":
        code = "other"
    result = "Movimento executado pelo sujeito: " +  code  
    
    return result


# Sends the result to the client
@socketio.on('process')
def handle_process():
    
    df = pd.read_csv("new_data.csv", names=['timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'])
    print(df)
    package = process(df)
    print(package)
    print("entrou handle process")
    emit('serverProcessResponse', {"class" : package })



@socketio.on('sendData')
def handle_my_custom_event(json):
    global thread
    print('Client connected')

    # Start the thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(event_stream)


if __name__ == '__main__':
    socketio.run(app, debug=True)
