import base64
import sys
import json

import cv2
import zmq

# connection for streaming video
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://localhost:8000')

camera = cv2.VideoCapture(sys.argv[1])
# frames_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frames_count)
frames_count = 0

while True:
    try:
        grabbed, frame = camera.read()  # grab the current frame
        if(frames_count == 3268):
            camera = cv2.VideoCapture(sys.argv[1])
            grabbed, frame = camera.read()
            frames_count = 0
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)
        frames_count += 1
    except KeyboardInterrupt:
        camera.release()
        cv2.destroyAllWindows()
        break