import random

from flask import Flask

from flask import request, jsonify
from flask import abort
from flask import render_template, redirect, url_for

from pymongo import MongoClient
from bson.objectid import ObjectId

# Define host name
sever_name="haai.cs.ualberta.ca:5000"

# Connect with database
database_url = "localhost:27017"
database = MongoClient(database_url)

experiments_col = database["user_study_test"]["experiments"]
results_col = database["user_study_test"]["results"]

app = Flask(__name__)

def get_min_experiments(experiment_counts):
    min_key = min(experiment_counts, key=experiment_counts.get)
    min_count = experiment_counts[min_key]

    min_experiment_ids = [experiment_id for experiment_id,count in experiment_counts.items() if min_count == count]
    return min_experiment_ids

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/')
def index():
    experiments = experiments_col.find({})
    experiments_to_evaluate = []

    experiment_counts = {}
    for e in experiments:
        experiment_counts[e['_id']] = results_col.count_documents({'experiment_id': e['_id']})

    # Get random min expertiment
    min_experiment_ids = get_min_experiments(experiment_counts)
    min_id = random.choice(min_experiment_ids)

    min_experiment = experiments_col.find_one({"_id": min_id})

    pieces = []
    for key in min_experiment:
        if key != '_id' and key != 'system':
            pieces.append(key)

    # Randomize pieces order
    random.shuffle(pieces)

    return render_template('index.html', experiment=min_experiment,
                                         order=pieces,
                                         sever_name=sever_name)

@app.route('/evaluate/<experiment_id>/<piece>')
def evaluate(experiment_id, piece):
    experiment = experiments_col.find_one({"_id": experiment_id})
    return render_template('evaluate.html', piece=experiment[piece],
                                      sever_name=sever_name)

@app.route('/test/<test_id>')
def test(test_id):
    if int(test_id) == 1:
        return render_template('test.html', piece='static/test/e2_real_human_4.mp3',
                                            q1=1, q2=5, q3=5,
                                            sever_name=sever_name)
    elif int(test_id) == 2:
        return render_template('test.html', piece='static/test/e4_real_human_2.mp3',
                                            q1=5, q2=2, q3=5,
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
                if key != '_id' and key != 'system':
                    result[key + "_q1"] = request.form.get(key + "_q1")
                    result[key + "_q2"] = request.form.get(key + "_q2")
                    result[key + "_q3"] = request.form.get(key + "_q3")
                    result[key + "_q4"] = request.form.get(key + "_q4")
                    result[key + "_q5"] = request.form.get(key + "_q5")
                    result[key + "_expl"] = request.form.get(key + "_expl")

            for test in range(1, 3):
                    result["test_{}_q1".format(test)] = request.form.get("test_{}_q1".format(test))
                    result["test_{}_q2".format(test)] = request.form.get("test_{}_q2".format(test))
                    result["test_{}_q3".format(test)] = request.form.get("test_{}_q3".format(test))
                    result["test_{}_q4".format(test)] = request.form.get("test_{}_q4".format(test))
                    result["test_{}_q5".format(test)] = request.form.get("test_{}_q5".format(test))

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
