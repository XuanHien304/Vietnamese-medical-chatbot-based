import torch
import json
import os
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for, flash
from pyserini.search.lucene import LuceneSearcher
from src.model import PhoBERTChatBot
from src.utils import problem_response, get_label, disease_response, chatgpt_response

searcher = LuceneSearcher('lookup_db')
searcher.set_language('vn')
searcher.set_bm25()

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.static_folder = 'static'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database', 'patients.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'random string'
app.app_context().push()

db = SQLAlchemy(app)

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

rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
with open('./data/intent_train.json', 'r') as json_data:
    contents = json.load(json_data)

model = PhoBERTChatBot('vinai/phobert-base', 8)
model.load_state_dict(torch.load('weight/saved_weights.pth',  map_location=torch.device('cpu')))
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
tags_set, contents = get_label('./data/intent_train.json')

@app.route('/delete/<int:id>')
def delete(id):
    id_delete = Patients.query.get_or_404(id)
    db.session.delete(id_delete)
    db.session.commit()
    return redirect(url_for('database'))

@app.route('/')
def home():   
    return render_template('chatbot.html')

@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    mode = request.args.get('mode')
    # answer, _ = chatbot_response(userText)
    answer, _ = disease_response(model, tokenizer, userText, rdrsegmenter, tags_set, contents)
    if mode == 'problem':
        if answer == 'Dạ bạn cho mình xin họ và tên ạ':
            return redirect(url_for('form'))
        if answer.startswith('bạn có thể') and len(userText.split(',')) < 3:
            return 'mình chưa rõ lắm, bạn có thể cho mình xin thêm thông tin về vấn đề bạn đang gặp phải không ạ'
        return answer
    elif mode == 'thongtin':
        return problem_response(searcher, userText)
    elif mode == 'chatgpt':
        return chatgpt_response(userText)

@app.route('/database')
def database():
    return render_template('database.html', results=Patients.query.all())

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        name = request.form['name']
        sex = request.form['sex']
        age = request.form['age']
        phone = request.form['phone']
        date = request.form['date']

        patient = Patients(name, sex, age, phone, date)
        db.session.add(patient)
        db.session.commit()
        flash('Sucessfully')
        return redirect(url_for('database'))
    return render_template('form.html')
    
if __name__ == "__main__":
    db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=False)
