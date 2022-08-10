import os
import csv
import argparse
import numpy as np

from pymongo import MongoClient
from bson.objectid import ObjectId

# Define host name
sever_name="localhost:5000"

# Connect with database
database_url = "localhost:27017"
database = MongoClient(database_url)

experiments_col = database["user_study_test"]["experiments"]
results_col = database["user_study_test"]["results"]

tests = {"static/test/e2_real_human_4.mp3": "test_1",
         "static/test/e4_real_human_2.mp3": "test_2"}


pieces = ('piece_1', 'piece_2', 'piece_3', 'piece_4')

systems = {
        'mcts' : {'v1': [], 'v0': [], 'a1': [], 'a0': [], 'h': [], 'r':[], 'o': []}, 
        'sbbs' : {'v1': [], 'v0': [], 'a1': [], 'a0': [], 'h': [], 'r':[], 'o': []}, 
        'human': {'v1': [], 'v0': [], 'a1': [], 'a0': [], 'h': [], 'r':[], 'o': []}, 
        'remi' : {'v1': [], 'v0': [], 'a1': [], 'a0': [], 'h': [], 'r':[], 'o': []}, 
}

profile = {
        'ethnicity': [],
        'language' : [],
        'age'      : [],
        'gender'   : [],
        'xp'       : [],
        'comments' : []
}

def is_valid_experiment(experiment, results):
    test_id = tests[experiment['piece_t']]

    test_q1 = int(results['{}_{}'.format(test_id, 'q1')]) 
    test_q2 = int(results['{}_{}'.format(test_id, 'q2')]) 
    test_q3 = int(results['{}_{}'.format(test_id, 'q3')]) 
    test_q4 = int(results['{}_{}'.format(test_id, 'q4')]) 
    test_q5 = int(results['{}_{}'.format(test_id, 'q5')]) 
    
    piecet_q1 = int(results['piece_t_{}'.format('q1')]) 
    piecet_q2 = int(results['piece_t_{}'.format('q2')]) 
    piecet_q3 = int(results['piece_t_{}'.format('q3')]) 
    piecet_q4 = int(results['piece_t_{}'.format('q4')]) 
    piecet_q5 = int(results['piece_t_{}'.format('q5')]) 

    q1_hit = False 
    if (test_q1 > 3 and piecet_q1 > 3) or (test_q1 < 3 and  piecet_q1 < 3):
        q1_hit = True

    q2_hit = False 
    if (test_q2 > 3 and piecet_q2 > 3) or (test_q2 < 3 and  piecet_q2 < 3):
        q2_hit = True

    q3_hit = False 
    if (test_q3 > 3 and piecet_q3 > 3) or (test_q3 < 3 and  piecet_q3 < 3):
        q3_hit = True

    explanations = []
    for p in pieces:
        expl = results['{}_expl'.format(p)]
        explanations.append(expl)

    explanations_validation = len(set(explanations)) == len(pieces)
    return (q1_hit and q2_hit and q3_hit) and explanations_validation

if __name__ == "__main__":
    results = results_col.find({})
    experiments = experiments_col.find({})

    total_valid = 0
    total_experiments = 0

    for e in experiments:
        experiment_results = results_col.find({'experiment_id': e['_id']})
        experiment_results_count = results_col.count_documents({'experiment_id': e['_id']})
        
        valid_results = 0
        for r in experiment_results:
            try:
                if not is_valid_experiment(e, r):
                    continue
            except:
                continue

            # Get profile data
            profile['ethnicity'].append(r['ethnicity'])
            profile['language'].append(r['language'])
            profile['age'].append(2022 - int(r['year']))
            profile['xp'].append(int(r['xp']))
            profile['comments'].append(r['comments'])

            for p in pieces:
                piece_emotion = os.path.basename(e[p]).split('.')[0].split('_')[0]
                piece_system = os.path.basename(e[p]).split('.')[0].split('_')[2]
                
                try:
                    q1 = int(r['{}_q1'.format(p)])
                    q2 = int(r['{}_q2'.format(p)])

                    # high valence
                    if piece_emotion  == 'e1' or piece_emotion == 'e4':
                        systems[piece_system]['v1'].append(q1)

                    # low valence
                    if piece_emotion == 'e2' or piece_emotion  == 'e3':
                        systems[piece_system]['v0'].append(q1)

                    # high arousal
                    if piece_emotion == 'e1' or piece_emotion == 'e2':
                        systems[piece_system]['a1'].append(q2)

                    # low arousal
                    if piece_emotion  == 'e3' or piece_emotion == 'e4':
                        systems[piece_system]['a0'].append(q2)

                    q3 = int(r['{}_q3'.format(p)])
                    q4 = int(r['{}_q4'.format(p)])
                    q5 = int(r['{}_q5'.format(p)])
                    
                    systems[piece_system]['h'].append(q3)
                    systems[piece_system]['r'].append(q4)
                    systems[piece_system]['o'].append(q5)
                    
                except:
                    print('Invalid experiment')

            valid_results += 1

        print('> Experiment {}: {}/{}'.format(e['_id'], valid_results, experiment_results_count))
    
        total_valid += valid_results
        total_experiments += experiment_results_count

    print("Experimentns (valid,total):", total_valid, total_experiments)

def save_field(systems, field, filename):
    max_len = max(len(systems['mcts'][field]), 
                  len(systems['human'][field]), 
                  len(systems['sbbs'][field]), 
                  len(systems['remi'][field]))
    
    header = ['human', 'mcts', 'sbbs', 'remi']
    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
    
        # write the header
        writer.writerow(header)
    
        row = []
        for i in range(max_len):
            mcts_i = None
            if i < len(systems['mcts'][field]):
                mcts_i = systems['mcts'][field][i]
            
            human_i = None
            if i < len(systems['human'][field]):
                human_i = systems['human'][field][i]
    
            sbbs_i = None
            if i < len(systems['sbbs'][field]):
                sbbs_i = systems['sbbs'][field][i]
    
            remi_i = None
            if i < len(systems['remi'][field]):
                remi_i = systems['remi'][field][i]
    
            row = [human_i, mcts_i, sbbs_i, remi_i] 
    
            writer.writerow(row)

save_field(systems, 'v1', 'high_valence.csv')
save_field(systems, 'v0', 'low_valence.csv')
save_field(systems, 'a1', 'high_arousal.csv')
save_field(systems, 'a0', 'low_arousal.csv')
save_field(systems, 'h', 'humaness.csv')
save_field(systems, 'r', 'richness.csv')
save_field(systems, 'o', 'overall_quality.csv')

print('Profile:')
print('> Average Age:', np.mean(profile['age']), np.std(profile['age']))
print('> Average Music Experience:', np.mean(profile['xp']), np.std(profile['xp']))
