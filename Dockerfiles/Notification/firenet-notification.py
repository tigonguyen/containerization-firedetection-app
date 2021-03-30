import base64
import numpy as np

import flask
from flask import request
import cv2
import telepot

TOKEN = "1167994481:AAESJVV4fUGGWk-eKo8GypwNJZ2iMERHnMQ"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
CHAT_ID = '721339255'

width = 1280
height = 720

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def mydefault():
    return "Success!"

@app.route('/notification', methods=['POST'])
def notification():
    jpg_as_text = request.get_data()
    # convert data from byte to image file
    img = base64.b64decode(jpg_as_text)
    npimg = np.fromstring(img, dtype=np.uint8)
    frame = cv2.imdecode(npimg, 1)

    cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
    cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
        cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)
    cv2.imwrite('fire.jpg', frame)
    
    bot = telepot.Bot(TOKEN)
    bot.sendPhoto(chat_id=CHAT_ID, photo=open('./fire.jpg', 'rb'), caption='Fire has just been detected on your camera!')

    return "Success!"

app.run(port=8080, host='0.0.0.0')
