import os
import json
import math
import base64
import numpy as np

import requests
import cv2
import zmq
import tflearn

from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

if __name__ == '__main__':
    # declaration for video streaming connection
    context = zmq.Context()
    footage_socket = context.socket(zmq.SUB)
    footage_socket.bind('tcp://*:8000')
    footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    # frame sizes
    width = 1280
    height = 720
    rows = 224
    cols = 224

    # construct and display model
    model = construct_firenet (224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")

    # window for displaying video
    cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
    
    frame_time = 33

    # var for couting frames for raising alert
    count_frame = 0

    keepProcessing = True
    
    while (keepProcessing):
        try:
            # start a timer (to see how long processing and display takes)
            start_t = cv2.getTickCount();
        
            # receive message and translate to frame
            jpg_as_text = footage_socket.recv_string()
            img = base64.b64decode(jpg_as_text)
            npimg = np.fromstring(img, dtype=np.uint8)
            frame = cv2.imdecode(npimg, 1)

            # predicion for fire in frame or not
            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
            output = model.predict([small_frame])

             # stop the timer and convert to ms. (to see how long processing and display takes)
            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            # displaying detecting results on image 
            if round(output[0][0]) == 1:
                count_frame += 1
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            else:
                count_frame = 0
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            

            # speed is about 12 frames/second, so 10 second have 120 frames
            if count_frame == 120:
                req = requests.post('http://localhost/notification', data=jpg_as_text)
                print("The starus code of request is:", req.status_code)
                print(req.raise_for_status())
                count_frame = 0

            # displaying video 
            cv2.imshow("Stream", frame)
            
            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x')):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
 