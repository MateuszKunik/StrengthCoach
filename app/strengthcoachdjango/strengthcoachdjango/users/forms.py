from allauth.account.forms import SignupForm
from allauth.socialaccount.forms import SignupForm as SocialSignupForm
from bootstrap_datepicker_plus.widgets import DatePickerInput
from django.contrib.auth import forms as admin_forms
from django.contrib.auth import get_user_model
from django.core.validators import MaxValueValidator, MinValueValidator
from django.forms import CharField, ChoiceField, DateTimeField, DecimalField, EmailField, IntegerField, ModelForm
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class UserProfileUpdateForm(ModelForm):
    """
    Form for updating user profile information.
    """

    first_name = CharField(max_length=255, label=_("First name of User"), required=True)
    last_name = CharField(max_length=255, label=_("Last name of User"), required=True)
    birth_date = DateTimeField(
        widget=DatePickerInput(options={"format": "DD-MM-YYYY"}),
        label=_("Date of birth"),
        input_formats=["%d-%m-%Y"],
        required=True,
    )
    gender = ChoiceField(choices=User.GENDER_CHOICES, label=_("Gender"), required=True)
    height = IntegerField(label=_("Height"), required=True, validators=[MinValueValidator(1), MaxValueValidator(250)])
    weight = DecimalField(
        label=_("Weight"),
        required=True,
        validators=[MinValueValidator(1), MaxValueValidator(250)],
        decimal_places=1,
        max_digits=5,
    )
    level = ChoiceField(choices=User.LEVEL_CHOICES, label=_("Proficiency level"), required=True)
    powerlifting_section = CharField(max_length=255, label=_("Powerlifting section"), required=False)
    months_of_experience = IntegerField(
        label=_("Number of months of experience"),
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(250)],
    )

    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "birth_date",
            "gender",
            "height",
            "weight",
            "level",
            "powerlifting_section",
            "months_of_experience",
        ]


class UserAdminChangeForm(admin_forms.UserChangeForm):
    class Meta(admin_forms.UserChangeForm.Meta):
        model = User
        field_classes = {"email": EmailField}


class UserAdminCreationForm(admin_forms.UserCreationForm):
    """
    Form for User Creation in the Admin Area.
    To change user signup, see UserSignupForm and UserSocialSignupForm.
    """

    class Meta(admin_forms.UserCreationForm.Meta):
        model = User
        fields = ("email",)
        field_classes = {"email": EmailField}
        error_messages = {
            "email": {"unique": _("This email has already been taken.")},
        }


class UserSignupForm(SignupForm):
    """
    Form that will be rendered on a user sign up section/screen.
    Default fields will be added automatically.
    Check UserSocialSignupForm for accounts created from social.
    """


class UserSocialSignupForm(SocialSignupForm):
    """
    Renders the form when user has signed up using social accounts.
    Default fields will be added automatically.
    See UserSignupForm otherwise.
    """
