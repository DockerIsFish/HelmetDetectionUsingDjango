from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import threading
import subprocess
import os



'''
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed,self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        SOURCE = self.frame
        _, jpeg = cv2.imencode('.jpg', SOURCE)


        return jpeg.tobytes()
    
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n' b'Content-Type: image/jped\r\n\r\n' + frame +b'\r\n\e\n')

        
@gzip.gzip_page
def RealTime(request):
    try:
        subprocess.run(['python3','detect.py','--source','0','--weights','best_50.pt'])
    except:
        print('Error')
        pass

'''
def RealTime(request):
    subprocess.run(['python3','detect.py','--source','0','--weights','best_50.pt'])
    return render(request,'index.html')


def index(request):
    return render(request, 'index.html')