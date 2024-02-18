from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render, reverse
from django.utils.translation import gettext_lazy as _
from moviepy.editor import VideoFileClip

from .forms import VideoUploadForm


@login_required
def upload_video(request):
    if not request.user.terms_confirmed:
        messages.warning(
            request,
            _(
                "Please complete your profile information and accept "
                "the terms and conditions before uploading your video."
            ),
        )
        return redirect("users:update")

    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)

        if form.is_valid():
            video = form.save(commit=False)
            video.user = request.user
            # time.sleep(10)
            video.save()

            video_path = video.video.path
            video_clip = VideoFileClip(video_path)
            video_length = video_clip.duration
            video_size = video_clip.size

            print(f"Video Length: {video_length} seconds")
            print(f"Video Size: {video_size} bytes")

            user_detail_url = reverse("users:detail", kwargs={"pk": request.user.pk})
            messages.success(request, _("The video has been successfully uploaded."))
            return redirect(user_detail_url)
    else:
        form = VideoUploadForm()
    return render(request, "workouts/upload_video.html", {"form": form})
