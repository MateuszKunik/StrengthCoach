from django.urls import path

from .views import upload_video

app_name = "workouts"
urlpatterns = [
    path("upload/", upload_video, name="upload_video"),
]
