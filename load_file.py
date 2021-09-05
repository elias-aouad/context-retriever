import json


def load_file(file_path):
    dataset = {"question":[], "title":[], "answer":[], "passage":[]}

    with open(file_path, "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        for key in dataset.keys():
            dataset[key] += [result[key]]
    return dataset
