# This is our own code
import os
import secrets
from PIL import Image
from flask import current_app


alpha = "abcdefghijklmnopqrstuvwxyz"
def savepicture(form_picture,code,message=""):
    #generating a random hex to store the image name and keeping its extenion as jpg
    random_hex = secrets.token_hex(5)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + '.jpg'
    message_fn  = random_hex + '.txt'
 

    #storing the path of the image
    if(code == 1):  
        picture_path = os.path.join(current_app.root_path, 'static/users/steganography', picture_fn)
        message_path = os.path.join(current_app.root_path, 'static/users/steganography', message_fn)
        print(message_path)
        i = Image.open(form_picture)
        #saving the image
        i.save(picture_path)
        #saving the message
        with open(message_path,'w') as f:
            f.write(message)
        #returning the picture name

        
        return picture_fn,1,message_fn
    else:
        picture_path = os.path.join(current_app.root_path, 'static/Test', picture_fn)

    i = Image.open(form_picture)
    #saving the image
    i.save(picture_path)
    #returninh the picture name
    return picture_fn, 1

