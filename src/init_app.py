import os
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

def initial_app():
    current_dir = os.getcwd()
    app = Flask(__name__)
    app.static_folder = 'static'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(current_dir, 'database/patients.sqlite3')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'random strin'
    app.app_context().push()
    return app

db = SQLAlchemy()

class Patients(db.Model):
    id = db.Column('id', db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    sex = db.Column(db.String(50))
    age = db.Column(db.Integer)
    diagnosis = db.Column(db.String(1000))
    date = db.Column(db.String(100))

    def __init__(self, name, sex, age, diagnosis, date):
        self.name = name
        self.sex = sex
        self.age = age
        self.diagnosis = diagnosis
        self.date = date