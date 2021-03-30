import base64
import sys
import json

import cv2
import zmq

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# connection for streaming video
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://192.168.1.2:8000')

camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
# frames_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frames_count)
frames_count = 0
frame_w = 224
frame_h = 224

cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)

while True:
    try:
        grabbed, frame = camera.read()  # grab the current frame
        #frame = cv2.resize(big_frame, (frame_w, frame_h), cv2.INTER_AREA)
        if(frames_count == 3268):
            camera = cv2.VideoCapture(sys.argv[1])
            grabbed, frame = camera.read()
            frames_count = 0
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)
        #frames_count += 1
        # displaying video 
        cv2.imshow("Stream", frame) 
        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(1)
    except KeyboardInterrupt:
        camera.release()
        cv2.destroyAllWindows()
        break
