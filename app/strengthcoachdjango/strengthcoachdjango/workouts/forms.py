from django import forms
from django.utils.translation import gettext_lazy as _

from .models import UserVideo


class VideoUploadForm(forms.ModelForm):
    video = forms.FileField(
        label=_("Upload video file"),
        required=True,
    )

    class Meta:
        model = UserVideo
        fields = ["video"]
