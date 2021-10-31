import os
import argparse
import json
import glob
import numpy as np
import pandas as pd

# Tags to specify sentence start and sentence end
# Теги для указания начала и конца предложения
SENTENCE_START_TAG = "S"
SENTENCE_PRESTART_TAG = "*"
SENTENCE_END_TAG = "E"

# function for calculating the part-of-speech transition probabilities
# функция для вычисления вероятности переходов частей речи
def t_given_uv(t, u, v, train_bag):
    tags = [
        pair[1] for pair in train_bag
    ]  # only tags are selected // выбераем только теги
    count_uv = 0
    count_t_uv = 0
    for index in range(len(tags) - 1):
        if tags[index] == u and tags[index + 1] == v:
            count_uv += 1
    for index in range(len(tags) - 2):
        if tags[index] == u and tags[index + 1] == v and tags[index + 2] == t:
            count_t_uv += 1
    return (count_t_uv, count_uv)



if __name__ == "__main__":
    # input command
    # анализ входной команды
    parser = argparse.ArgumentParser(description="A part-of-speech Hidden Markov Model trainer")
    parser.add_argument("-r", default=None, type=str, dest="readFrom", help="Full path to the folder containing the training data in json format (it looks into /datasets if the folder is not specified)")
    parser.add_argument("-w", default=None, type=str, dest="writeTo", help="Full path to the folder to save the data obtained from the training set (it saves into /data if the folder is not specified)")
    args = parser.parse_args()

    # gets the project's working folder
    # рабочую папку проекта 
    working_folder = "{}".format(os.getcwd())

    # folder with the training data in json format
    # папка с обучающими данными в формате json 
    dataset_folder = working_folder + "/datasets/" if args.readFrom == None else "{}".format(args.readFrom)
    isExist = os.path.exists(dataset_folder)
    if not isExist:
        raise ValueError("The specified path to the folder containing the json files with the training data does not exists.")

 
    # folder to save the obtained data from the training process
    # папка для загрузки данных из процесса обучения 
    data_folder = working_folder + "/data/" if args.writeTo == None else "{}".format(args.writeTo)
    isExist = os.path.exists(data_folder)
    if not isExist:
        raise ValueError("The specified path to the folder to save the obtained data does not exists. Please, create it manually first.")


    # a list for temporary storage of selected datasets // список для временного хранения выбранных наборов данных   
    training_data = []        

    # load json files // загружается файлы json
    print('Uploading json files with the training data...')
    all_files = glob.glob(dataset_folder + '*.json')
    for filename in all_files:
        user_response = False
        valid_responses = ['y', 'n']
        while user_response == False:
            response = input('Use dataset {}? [y/n]'.format(filename.split('/')[-1]))
            if response in valid_responses:
                if response == 'y':
                    try:
                        with open(filename) as json_file:
                            data = json.load(json_file)
                        training_data.append(data)
                    except Exception as e:
                        print(e)
                    else:
                        print("Training dataset loaded")
                user_response = True
            else:
                print('Response not valid')

    # total of selected datasets
    # количество выбранных наборов данных   
    total_seleted_datasets = len(training_data)
    if total_seleted_datasets < 1:
        print('No datasets selected. Exiting script...')
        exit()

    # extracts the pairs worf-form and part-of-speech tags from the training set of sentences
    # извлекает словоформы и части речи из обучающего набора
    word_postag_pairs = []
    postags = {}    # dictionary with totals for each part-of-speech tag / словарь с итогами по каждому тегу части речи
    total_postags = 0       # total part-of-speech tags / общее количество тегов части речи
    for x in training_data:
        content = x['content']
        for sentence in content:
            annotations = sentence['parse']
            l = [(SENTENCE_PRESTART_TAG, SENTENCE_PRESTART_TAG), (SENTENCE_START_TAG, SENTENCE_START_TAG)]
            for i in annotations:
                if i['postag'] != 'PNCT':
                    if i["postag"] not in postags.keys():
                        postags[i['postag']] = 0
                    postags[i['postag']] += 1
                    total_postags += 1
                    l.append((i['token'], i['postag']))
            l.append((SENTENCE_END_TAG, SENTENCE_END_TAG))
            word_postag_pairs.append(l)

    # extracts the distribution of part-of-speechs in the training set
    # извлекает распределение частей речи в обучающем наборе
    postag_dist = {}
    for k, v in postags.items():
        postag_dist[k] = v / total_postags

    
    # saves the extracted part-of-speech distribution to a json file
    # сохраняет извлеченное распределение частей речи в файл json
    with open(data_folder + "postag_distribution.json", "w") as outfile:
        json.dump(postags, outfile, indent=4, sort_keys=True)


    # list of tuples // список кортежов
    train_tagged_words = [tup for sent in word_postag_pairs for tup in sent]
    # set with unique tags form the training set // уникальные теги в обучающих данных
    tags = {tag for word, tag in train_tagged_words}


    # creates a dictionary for storing the frequency word-forms under a specific part-of-speech tag
    # создает словарь для хранения частотных словоформ под определенной частью речи
    dictionary = {}
    for tup in train_tagged_words:
        token = tup[0].lower()
        if token not in ["*", "s", "e"]:
            postag = tup[1]
            if postag not in dictionary.keys():
                dictionary[postag] = {}
            if token not in dictionary[postag].keys():
                dictionary[postag][token] = 0
            dictionary[postag][token] += 1

    # сalculates the emission probabilities for each part-of-speech tag
    # вычисляет вероятности результата для каждого тега части речи
    emission_probabilities = {}
    for k, v in dictionary.items():
        for k2, v2 in v.items():
            if k2 not in emission_probabilities.keys():
                emission_probabilities[k2] = {}
            if k not in emission_probabilities[k2].keys():
                emission_probabilities[k2][k] = 0
            emission_probabilities[k2][k] = v2 / postags[k]
            dictionary[k][k2] = v2 / postags[k]


    # creates a json file to store the emission probabilities
    # создает файл json для хранения вероятностей результата
    with open(data_folder + "emission_probabilities.json", "w") as outfile:
        json.dump(emission_probabilities, outfile, indent=4, sort_keys=True)


    # prepares the matrix to store the transition emission probabilities
    # подготавливает матрицу для хранения вероятностей переходов
    combined_tags = ["{}_{}".format(SENTENCE_PRESTART_TAG, SENTENCE_START_TAG)]
    for u in list(tags):
        if u not in [SENTENCE_PRESTART_TAG, SENTENCE_END_TAG]:
            for v in list(tags):
                if v not in [SENTENCE_PRESTART_TAG, SENTENCE_START_TAG, SENTENCE_END_TAG]:
                    comb_tag = "{}_{}".format(u, v)
                    combined_tags.append(comb_tag)
    tags.remove(SENTENCE_START_TAG)
    tags.remove(SENTENCE_PRESTART_TAG)
    tags_matrix = np.zeros((len(combined_tags), len(tags)), dtype="float32")


    # training algorithm for calculating the transition probabilities
    # обучающий алгоритм расчета вероятностей переходов
    cp_counter = 1
    for i, u_v in enumerate(combined_tags):
        for j, t in enumerate(list(tags)):
            uv = u_v.split("_")
            try:
                probability = t_given_uv(t, uv[0], uv[1], train_tagged_words)[0] / (t_given_uv(t, uv[0], uv[1], train_tagged_words)[1])
            except:
                probability = 0
            tags_matrix[i, j] = probability
            print("[{}] t:{} / u:{} x v:{} = {}".format(cp_counter, t, uv[0], uv[1], probability))
            cp_counter += 1
            if probability > 0:
                tags_matrix[i, j] = probability
            else:
                tags_matrix[i, j] = 0.000000001  # a very small probability for non existing combinations // минимальная вероятность для несуществующих комбинаций 
            tags_df = pd.DataFrame(tags_matrix, columns=list(tags), index=combined_tags)

    
    # creates the files with the probabilities
    # создаются файлы с вероятностей переходов
    tags_df.index = combined_tags
    print(tags_matrix.shape)
    print(tags_df)
    df1_transposed = tags_df.T
    df_sorted = df1_transposed.sort_index(axis=1)
    df_sorted_index = df_sorted.sort_index()
    df_sorted_index.to_csv(data_folder + "transition_probabilities.csv")
    tags_df.to_json(data_folder + "transition_probabilities.json", indent=4)

    print("The training process is finished.")

