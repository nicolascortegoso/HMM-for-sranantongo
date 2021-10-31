import os
import json
import argparse
import csv
from sklearn.metrics import multilabel_confusion_matrix
from tagger import Tokenizer, Emission, Transition


def divide(numerator, denominator):
    denom = numerator + denominator
    if denom == 0:
        return 0
    else:
        return round(numerator / denom, 2)

# f-score
def f_1(precision, recall):
    numerator = 2 * (precision * recall)
    denominator = precision + recall
    try:
        return round(numerator / denominator, 2)
    except:
        return 0


def calculate_precision_recall_f(filename, conf_matrix):
    with open(filename, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Tег", "TN", "FP (I err)", "FN (II err)", "TP", "Precision", "Recall", "F-score",])
        precision_list = []
        recall_list = []
        f_list = []
        for i, matrix in enumerate(conf_matrix):
            tp = int(matrix[1][1])
            fp = int(matrix[0][1])
            fn = int(matrix[1][0])
            precision = divide(tp, fp)
            precision_list.append(precision)
            recall = divide(tp, fn)
            recall_list.append(recall)
            f = f_1(precision, recall)
            f_list.append(f)
            writer.writerow([postag_list[i], matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1], precision, recall, f])
        average_precision = round(sum(precision_list) / len(precision_list), 2)
        average_recall = round(sum(recall_list) / len(recall_list), 2)
        average_f = round(sum(f_list) / len(f_list), 2)
        writer.writerow(["Average", "", "", "", "", average_precision, average_recall, average_f])


if __name__ == "__main__":
    # get working path
    # рабочий путь
    working_folder = "{}".format(os.getcwd())

    parser = argparse.ArgumentParser(description="Script for testing the part-of-speech tagger")
    parser.add_argument("filename", type=str, help="Full path to the json formated file containing the testing sentences")
    parser.add_argument("-r", default=None, type=str, dest="readFrom", help="Full path to the folder that contains the statistical data obtained from the training set (it looks into /data if the folder is not specified)")
    parser.add_argument("-w", default=None, type=str, dest="writeTo", help="Specifies the path and filename with the extension .csv to save the test results")
    args = parser.parse_args()

    # Fast check to detect proper file format
    # Быстрая проверка для определения правильного формата файла
    filename = args.filename
    if filename[-5:] != ".json":
        raise ValueError("The path does not point to a file with json format")

    # loads json file
    # загружает файл json
    try:
        with open(filename) as json_file:
            txt = json.load(json_file)
    except Exception as e:
        print(e)
        exit()
    else:
        print("Testing set loaded")


    # folder with the statistical data in json format
    # папка со статистическими данными в формате json 
    data_folder = working_folder + "/data/" if args.readFrom == None else "{}".format(args.readFrom)
    isExist = os.path.exists(data_folder)
    if not isExist:
        raise ValueError("The specified path to the folder containing the json files with the statistical data does not exists.")

  
    # file to save the results from the testing process
    # файл для хранения результатов теста
    results_file = args.writeTo
    if results_file != None:
        if results_file[-4:] != ".csv":
            raise ValueError("The file for storing the testing results must have a .csv extension")
    else:         
        results_file = 'results.csv'
    

    # required files
    # обязательных файла
    postag_dist = data_folder + "postag_distribution.json"
    emission_prob = data_folder + "emission_probabilities.json"
    transition_prob = data_folder + "transition_probabilities.json"

    # object initialization
    # инициализация объекта 
    tokenizer = Tokenizer()
    emission = Emission(postag_dist, emission_prob)
    transition = Transition(transition_prob)

    # part-of-speech tag lists (observed and predicted)
    # списки тегов части речи 
    true_postags = []
    predicted_postags = []
    postag_list = []

    # process for marking the text with part-of-speech tags
    # процесс разметки текста частями речи

    sentences = txt['content']
    for sentence in sentences:
        txt = sentence['srn']
        parse = sentence['parse']
        for p in parse:
            postag = p["postag"]
            if postag != "PNCT":
                true_postags.append(postag)
                if postag not in postag_list:
                    postag_list.append(postag)
        tokenized_sentences = tokenizer.tokenize(txt)
        for sentence in tokenized_sentences:
            tagged_tokens = emission.get_emission_probabilities(sentence)  # assigns the possible tags to a word form
            disambiguated_sequence = transition.get_sequence(tagged_tokens)  # disambiguates the assigned tags in the context of the sentence
            for token in disambiguated_sequence:
                predicted_tag = token[1]
                if predicted_tag != "PNCT":
                    predicted_postags.append(predicted_tag)

    # confusion matrix and testing results
    # матрица путаницы и результаты тестирования 
    confusion_matrix = multilabel_confusion_matrix(true_postags, predicted_postags, labels=postag_list)
    calculate_precision_recall_f(results_file, confusion_matrix)
    print('Done')
    print("Testing results saved to file: {}".format(results_file))

