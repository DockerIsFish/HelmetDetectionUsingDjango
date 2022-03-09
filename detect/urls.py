from django.urls import path
from . import views01, views02
app_name = 'detect'

urlpatterns = [
    path('',views01.home, name='home'),
    path('detect_face',views01.detect_face, name='detect_face'),
    path('next_page',views01.next_page, name='next_page'),
    path('detect_helmet',views02.detect_helmet, name='detect_helmet'),
]