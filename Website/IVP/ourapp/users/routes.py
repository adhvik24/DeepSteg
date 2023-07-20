# This part is our own original code

from flask import render_template, url_for, flash, redirect, request, Blueprint,current_app
from ourapp.users.forms import Steganography,Steganalysis
from ourapp.users.utils import savepicture
from ourapp.users.helper import predict
#Blueprint
users = Blueprint('users',__name__)
import os
from os import remove
import sys
sys.path.append('/home/arun/Desktop/Acads/Sem6/IVP/Project/IVP')
from time import sleep
from test import embed
import warnings
warnings.filterwarnings("ignore")

# Login Route 
@users.route('/',methods=['GET','POST'])
def home():
   
    return render_template('home.html',title='Home')


@users.route('/steganography',methods=['GET','POST'])
def steganography():
    form = Steganography()
    if(request.method=='POST'):
       if form.validate_on_submit():
            if form.picture.data and form.message.data:
                #saving the picture
                message = form.message.data
                picture_file,valid,message_file = savepicture(form.picture.data,1,message)
            if valid:
                 imagepath = os.path.join(current_app.root_path, 'static/users/steganography', picture_file)
                 messagepath = os.path.join(current_app.root_path, 'static/users/steganography', message_file)

            new_image_path = os.path.join(current_app.root_path,'static/users/steganalysis', 'new_'+picture_file)
           
            embed(imagepath,messagepath,new_image_path)
            new_image_path = url_for('static',filename='users/steganalysis/'+'new_'+picture_file)
            return render_template('Steganography.html',title='Steganography',form=form,image_file=new_image_path,yayfound = True)
    flash("Note: It takes around 1 min to embed the image after you submit","success")
    return render_template('Steganography.html',title='Steganography',form=form)



@users.route('/steganalysis',methods=['GET','POST'])
def steganalysis():
    form = Steganalysis()
    if(request.method=='POST'):
        if form.validate_on_submit():
            if form.picture.data: 
                picture_file,valid = savepicture(form.picture.data,0)
            if valid:
                imagepath = os.path.join(current_app.root_path, 'static/Test', picture_file)
                #code to wait for the image to be embedded
                sleep(2)
                
                prediction = predict()
           
                if(len(prediction)!=0):
                    flash(f"Prediction score is {prediction[-1]}","success")
                else:
                    flash("Please try again","danger")

                return render_template('Steganalysis.html',title='Steganography',form=form)

    return render_template('Steganalysis.html',title='Steganography',form=form)