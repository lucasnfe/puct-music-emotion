import os
import argparse
from pymongo import MongoClient

# Define host name
sever_name="localhost:5000"

# Connect with database
database_url = "localhost:27017"
database = MongoClient(database_url)

# Create database
mydb = database["user_study"]
experiments_col = database["user_study"]["experiments"]

def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):

    if verbose:
        print('[*] Scanning...')

    cnt, file_list = 0, []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue

                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')

    if is_sort:
        file_list.sort()

    return file_list

def retrieve_study(file_list):
    study = {}
    for f in audio_files:
        f_basename = os.path.splitext(os.path.basename(f))[0]

        experiment_emotion = f_basename.split('_')[0]
        experiment_system = f_basename.split('_')[1]
        experiment_index = f_basename.split('_')[3]

        experiment_key = (experiment_emotion, experiment_index)
        if experiment_key not in study:
            study[experiment_key] = {'_id': '{}_{}'.format(experiment_emotion, experiment_index),  'emotion': experiment_emotion, 'pieces': []}

        study[experiment_key]['pieces'].append(f)

    for experiment in study:
        for i, piece in enumerate(study[experiment]['pieces']):
            study[experiment]['piece_{}'.format(i+1)] = piece
        del study[experiment]['pieces']

    return study

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='pairwise.py')
    parser.add_argument('--path_indir', type=str, required=True, help="Path to input midi pieces.")
    args = parser.parse_args()

    audio_files = traverse_dir(
        args.path_indir,
        is_sort=True,
        extension='mp3')

    study = retrieve_study(audio_files)

    for experiment in study:
        experiments_col.insert_one(study[experiment])
