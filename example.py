import os
from tagger import Tokenizer, Emission, Transition

# example text
# пример текста
sample_text = 'Kofi lobi a umapikin.'

if __name__ == "__main__":
    # path to folder containing json files
    # путь к папке, содержащей файлы json
    root_folder = os.getcwd()

    # required json files
    # обязательных json файла
    data_folder = root_folder + "/data/"
    postag_dist = data_folder + "postag_distribution.json"
    emission_prob = data_folder + "emission_probabilities.json"
    transition_prob = data_folder + "transition_probabilities.json"

    # object initialization
    # инициализация объекта
    tokenizer = Tokenizer()
    emission = Emission(postag_dist, emission_prob)
    transition = Transition(transition_prob)

    # tokenization proceses (separates the text into a list of sentences and tokens )
    # процесс токенизации (разделяет текст в список предложений и токены)
    tokenized_sentences = tokenizer.tokenize(sample_text) 
    for sentence in tokenized_sentences:
        # process for marking the text with part-of-speech tags
        # процесс разметки текста частями речи
        tagged_tokens = emission.get_emission_probabilities(sentence)  # assigns the possible tags to a word form / присваивает возможные теги словоформе 
        disambiguated_sequence = transition.get_sequence(tagged_tokens)  # disambiguates the assigned tags in the context of the sentence
        print(disambiguated_sequence)
