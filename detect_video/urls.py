from django.urls import path
from . import views01

app_name = 'detect_video'

urlpatterns = [
    path('detect_video',views01.index, name='index'),
    path('stream',views01.RealTime, name='stream')
]