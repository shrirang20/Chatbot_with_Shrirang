from database import db
import datetime

class Prompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.String, nullable=False)
    answer = db.Column(db.String, nullable=False)
    time = db.Column(db.DateTime, nullable=False, default = datetime.datetime.utcnow)