from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)


app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

#db Config
app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:///site.db'

#Sql db init
db = SQLAlchemy(app)
db.app = app


#importing our packages
from ourapp.users.routes import users

#registering the blueprint for our packages
app.register_blueprint(users)




