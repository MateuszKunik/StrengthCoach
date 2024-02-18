from datetime import datetime

from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


def generate_filename(instance, filename):
    user_id = instance.user_id
    user_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"users_videos/user_{user_id}_{user_datetime}_{filename}"


class UserVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.FileField(upload_to=generate_filename, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user}'s Video {self.id}"

    class Meta:
        ordering = ["-created_at"]  # Default sorting in reverse order of creation date
