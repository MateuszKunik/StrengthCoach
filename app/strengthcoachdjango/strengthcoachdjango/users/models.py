from django.contrib.auth.models import AbstractUser
from django.db.models import (
    BooleanField,
    CharField,
    DateField,
    EmailField,
    FloatField,
    IntegerField,
    PositiveIntegerField,
)
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from strengthcoachdjango.users.managers import UserManager


class User(AbstractUser):
    """
    Default custom user model for StrengthCoachDjango.
    If adding fields that need to be filled at user signup,
    check forms.SignupForm and forms.SocialSignupForms accordingly.
    """

    GENDER_CHOICES = [
        ("M", _("Male")),
        ("F", _("Female")),
        ("O", _("Other")),
    ]

    LEVEL_CHOICES = [
        ("beginner", _("beginner")),
        ("intermediate", _("intermediate")),
        ("advanced", _("advanced")),
    ]

    # First and last name do not cover name patterns around the globe
    name = CharField(_("Name of User"), blank=True, max_length=255)
    first_name = CharField(_("First name of User"), max_length=255, null=True)
    last_name = CharField(_("Last name of User"), max_length=255, null=True)
    email = EmailField(_("email address"), unique=True)
    username = None  # type: ignore
    birth_date = DateField(_("Date of birth"), null=True)
    gender = CharField(_("Gender"), max_length=1, choices=GENDER_CHOICES, null=True)
    height = PositiveIntegerField(_("Height"), null=True)
    weight = FloatField(_("Weight"), null=True)
    level = CharField(_("Proficiency level"), max_length=20, choices=LEVEL_CHOICES, null=True)
    powerlifting_section = CharField(_("Powerlifting section"), max_length=50, null=True)
    months_of_experience = IntegerField(_("Number of months of experience"), null=True)
    terms_confirmed = BooleanField(null=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = UserManager()

    def get_absolute_url(self) -> str:
        """Get URL for user's detail view.

        Returns:
            str: URL for user detail.

        """
        return reverse("users:detail", kwargs={"pk": self.id})
