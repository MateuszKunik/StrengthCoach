from django.contrib import admin

from .models import UserVideo


@admin.register(UserVideo)
class UserVideoAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "display_filename")
    readonly_fields = ("id", "user", "created_at", "video")

    @admin.display(description="File Name")
    def display_filename(self, obj):
        return obj.video.name
