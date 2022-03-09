from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import numpy as np
# Create your views here.
def home(request):
    return render(request, 'home.html')

model = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = './deploy.prototxt'
net = cv2.dnn.readNet(model, config)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed,self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        blob = cv2.dnn.blobFromImage(self.frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detect = net.forward()

        detect = detect[0, 0, :, :]
        (h, w) = self.frame.shape[:2]
        
        number = 0
        
        for i in range(detect.shape[0]):
            confidence = detect[i, 2]
            
            if confidence < 0.5:
                break

            x1 = int(detect[i, 3] * w*0.6)
            y1 = int(detect[i, 4] * h*0.2)
            x2 = int(detect[i, 5] * w* 1.2)
            y2 = int(detect[i, 6] * h*1.5)
            face = self.frame[y1:y2,x1:x2]
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0))

            label = "Face : {}".format(number + 1)
            cv2.putText(self.frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            number = number + 1 
            cv2.imwrite('./static/images/jeonghoon1.jpg', face)
        #cv2.imshow('frame', self.frame)
        return jpeg.tobytes()
    
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
 
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n' b'Content-Type: image/jped\r\n\r\n' + frame +b'\r\n\e\n')


@gzip.gzip_page
def detect_face(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type = 'multipart/x-mixed-replace;boundary=frame')
    except:
        print('Error')
        pass

def next_page(request):
    return render(request,'image.html')


