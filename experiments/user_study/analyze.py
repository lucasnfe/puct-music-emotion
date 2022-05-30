import os
import argparse
import numpy as np

from pymongo import MongoClient
from bson.objectid import ObjectId

# Define host name
sever_name="localhost:5000"

# Connect with database
database_url = "localhost:27017"
database = MongoClient(database_url)

pairs_col = database["user_study"]["pairs_v2"]
quality_col = database["user_study"]["quality_v2"]

def legit_evaluation(evaluation, min_just_len=5):
    test_quality = int(evaluation["test_quality"])
    test_system1 = evaluation["test_piece1"].split("/")[-1].split(".")[0]
    test_system2 = evaluation["test_piece2"].split("/")[-1].split(".")[0]

    justification = evaluation["justification"]
    test_justification = evaluation["test_justification"]

    # Test cant be a tie
    if test_quality == 2:
        return False

    if test_system1 == "test_human" and test_quality > 2:
        return False

    if test_system2 == "test_human" and test_quality < 2:
        return False

    if len(test_justification.split(" ")) < min_just_len:
        return False

    if len(justification.split(" ")) < min_just_len:
        return False

    return True

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='analyze.py')
    # parser.add_argument('--midi', type=str, required=True, help="Path to midi data.")
    # parser.add_argument('--intro', type=str, required=True, help="Path to intro data.")
    opt = parser.parse_args()

    results_aggregate = {}
    results_between_systems = {}
    for evaluation in quality_col.find({}):
        # Retrieve pair data
        system1 = evaluation["piece1"].split("/")[3]
        system2 = evaluation["piece2"].split("/")[3]

        # Filter evaluation
        if not legit_evaluation(evaluation):
            continue

        # Process result
        quality = int(evaluation["quality"])

        if system1 not in results_aggregate:
            results_aggregate[system1] = {"wins": 0, "ties": 0, "losses": 0}

        if system2 not in results_aggregate:
            results_aggregate[system2] = {"wins": 0, "ties": 0, "losses": 0}

        if (system1, system2) not in results_between_systems:
            results_between_systems[(system1, system2)] = {"wins": 0, "ties": 0, "losses": 0}

        if quality == 2:
            results_aggregate[system1]["ties"] += 1
            results_aggregate[system2]["ties"] += 1
            results_between_systems[(system1, system2)]["ties"] += 1
        elif quality < 2:
            results_aggregate[system1]["wins"] += 1
            results_aggregate[system2]["losses"] += 1
            results_between_systems[(system1, system2)]["wins"] += 1
        elif quality > 2:
            results_aggregate[system1]["losses"] += 1
            results_aggregate[system2]["wins"] += 1
            results_between_systems[(system1, system2)]["losses"] += 1

    for system in results_aggregate:
        print(system, results_aggregate[system])

    print("====")
    for system_pair in results_between_systems:
        print(system_pair, results_between_systems[system_pair])

    combined_pairs = {}
    for system1 in results_aggregate:
        for system2 in results_aggregate:
            if (system1, system2) in results_between_systems:
                if (system1, system2) not in combined_pairs and (system2, system1) not in combined_pairs:
                    combined_pairs[(system1, system2)] = {"wins": 0, "ties": 0, "losses": 0}

                    scores1 = list(results_between_systems[(system1, system2)].values())
                    scores2 = list(results_between_systems[(system2, system1)].values())
                    total = np.array(scores1) + np.array(scores2[::-1])

                    combined_pairs[(system1, system2)]["wins"] = total[0]
                    combined_pairs[(system1, system2)]["ties"] = total[1]
                    combined_pairs[(system1, system2)]["losses"] = total[2]

    print("====")
    for pair in combined_pairs:
        print(pair, combined_pairs[pair])
