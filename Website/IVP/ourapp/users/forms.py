# This is our own code
from flask import Flask
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_login import current_user
# from ourapp.models import User


class Steganography(FlaskForm):
    message = StringField('Message To Be Encoded', validators=[Length(min=2,max=1000)])
    picture = FileField('Upload Picture', validators=[DataRequired(),FileAllowed(['jpg'])])
    submit = SubmitField('Submit')

class Steganalysis(FlaskForm):
    picture = FileField('Upload Picture', validators=[DataRequired(),FileAllowed(['jpg'])])
    submit = SubmitField('Submit')
