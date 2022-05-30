import random

from flask import Flask

from flask import request, jsonify
from flask import abort
from flask import render_template, redirect, url_for
from analyze import legit_evaluation

from pymongo import MongoClient
from bson.objectid import ObjectId

# Define host name
sever_name="172.31.81.219:4999"

# Connect with database
database_url = "localhost:27017"
database = MongoClient(database_url)

experiments_col = database["user_study"]["experiments"]
results_col = database["user_study"]["results"]

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/')
def index():
    experiments = experiments_col.find({})
    experiments_to_evaluate = []

    min_id = None
    min_count = float('inf')
    for e in experiments:
        count = results_col.count_documents({'experiment_id': e['_id']})
        if count < min_count:
            min_id = e['_id']
            min_count = count

    min_experiment = experiments_col.find_one({"_id": min_id})

    pieces = []
    for key in min_experiment:
        if key != '_id' and key != 'emotion':
            pieces.append(key)

    # Randomize pieces order
    random.shuffle(pieces)

    return render_template('index.html', experiment=min_experiment,
                                         order=pieces,
                                         sever_name=sever_name)

@app.route('/evaluate/<experiment_id>/<piece>')
def evaluate(experiment_id, piece):
    print(experiment_id)
    print(piece)
    experiment = experiments_col.find_one({"_id": experiment_id})
    return render_template('evaluate.html', piece=experiment[piece],
                                      sever_name=sever_name)

@app.route('/profile')
def profile():
    return render_template('profile.html', sever_name=sever_name)

@app.route('/end', methods = ['GET', 'POST'])
def end():
    if request.method == 'POST':
        result = {}
        if request.form.get('experiment'):
            experiment_id = request.form.get('experiment')
            experiment = experiments_col.find_one({"_id": experiment_id})

            result["experiment_id"] = experiment_id
            for key in experiment:
                if key != '_id' and key != 'emotion':
                    result[key + "_q1"] = request.form.get(key + "_q1")
                    result[key + "_q2"] = request.form.get(key + "_q2")
                    result[key + "_q3"] = request.form.get(key + "_q3")
                    result[key + "_q4"] = request.form.get(key + "_q4")
                    result[key + "_q5"] = request.form.get(key + "_q5")

            result["ethnicity"] = request.form.get("ethnicity")
            result["language"] = request.form.get("language")
            result["year"]     = request.form.get("year")
            result["xp"]       = request.form.get("xp")
            result["comments"] = request.form.get("comments")

            insert_result = results_col.insert_one(result);
            return redirect(url_for('end', evaluation_id=insert_result.inserted_id,
                                           sever_name=sever_name))
    else:
        return render_template('end.html', sever_name=sever_name)
