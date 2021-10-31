import json
import re

# emission probabilities / вероятности результата 
class Emission(object):
    """A class to assign probabilities to a word-form given a part-of-speech."""
    def __init__(self, postag_dist, emission_prob, default_tags = ["NN", "JJ", "RB", "VB", "ST"], propername_tags = ["NP"], number_tag = "NUMB", punctuation_tag = "PNCT"):
        self.openclass_tags = []
        self.propername_tags = []
        self.number_tag = number_tag
        self.punctuation_tag = punctuation_tag
        with open(postag_dist, "r") as postag_dist_json:
            self.postag_dist = json.load(postag_dist_json)
        with open(emission_prob, "r") as emission_prob_json:
            self.emission_prob = json.load(emission_prob_json)
        for postag in default_tags:
            if postag in self.postag_dist.keys() and postag not in self.openclass_tags:
                self.openclass_tags.append(postag)
        for postag in propername_tags:
            if postag in self.postag_dist.keys() and postag not in self.propername_tags:
                self.propername_tags.append(postag)
        self.D = 0
        for k, v in self.postag_dist.items():
            self.D += v

    def __get_postags(self, token):
        """Method to return the part-of-speech tags extracted from the training set."""
        word = token.lower()
        if word in self.emission_prob.keys():
            postags = self.emission_prob[word]
            return [(k, v) for k, v in postags.items()]
        return False

    def __estimate_postags(self, token, position, metric):
        """Method that estimates a probability for words not found in the training set according to the chosen metric."""
        methods = ["frec", "ln", "itf", "none"]
        estimated_tags = self.openclass_tags
        if token[0].isupper():
            if position == 0:
                estimated_tags = estimated_tags + self.propername_tags
            else:
                estimated_tags = self.propername_tags
        if metric in methods and len(estimated_tags) > 1:
            if metric == "itf":
                postags = self.__itf(estimated_tags)
            elif metric == "none":
                postags = [(tag, 1 / len(estimated_tags)) for tag in estimated_tags]
            elif metric == "frec" or metric == "ln":
                postags = self.__frec(estimated_tags, metric)
        else:
            postags = [(tag, (self.postag_dist[tag] / self.D)) for tag in estimated_tags]
        return postags

    def __frec(self, tag_list, metric):
        """Metric that translates (directly or logged) the proportion of tags in the training set."""
        emission_probabilities = {}
        total_frequencies = 0
        for tag in tag_list:
            frequency = self.postag_dist[tag]
            if metric == "ln":
                frequency = self.__ln(frequency) + 0.00000001
            total_frequencies += frequency
            emission_probabilities[tag] = frequency
        for k, v in emission_probabilities.items():
            emission_probabilities[k] = v / total_frequencies
        sorted_p = sorted(emission_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_p

    def __itf(self, tag_list):
        """Metric that penalizes frequency, making those tags with lower counts more likely."""
        emission_probabilities = {}
        total_frequencies = 0
        for tag in tag_list:
            frequency = self.postag_dist[tag]
            total_frequencies += frequency
            emission_probabilities[tag] = frequency
        for k, v in emission_probabilities.items():
            emission_probabilities[k] = (total_frequencies - v) / (total_frequencies * (len(emission_probabilities) - 1))
        sorted_p = sorted(emission_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_p

    def get_emission_probabilities(self, token_list, metric=None):
        """Main method to assign to a word given a part-of-speech. It calls other helper functions."""
        postags = []
        for position, token in enumerate(token_list):
            entry = token[0]
            if token[1] in ["word", "acronym"]:
                retreived_tags = self.__get_postags(entry)
                if retreived_tags is False:
                    retreived_tags = self.__estimate_postags(entry, position, metric)
                    t = (entry, retreived_tags)
                else:
                    t = (entry, retreived_tags)
            elif token[1] == "number":
                t = (entry, [(self.number_tag, 1)])
            else:
                t = (entry, [(self.punctuation_tag, 1)])
            postags.append(t)
        return postags

    def __ln(self, x):
        """A simple function to get the log of a value."""
        val = x
        return 99999999 * (x ** (1 / 99999999) - 1)


# Transition probabilities / вероятности результата
class Transition(object):
    """A class to assign probabilities to a part-ofword-form given a part-of-speech."""
    def __init__(self, transition_probabilities, punctuation_tag="PNCT"):
        self.punctuation_tag = punctuation_tag
        with open(transition_probabilities) as json_file:
            self.data = json.load(json_file)

    def __separate_punctuation_marks(self, sentence):
        """A helper method that removes the punctuation marks from token list and saves them with their possition to restore them after the process is finished."""
        sentence_without_punctuation_marks = []
        punctuation_marks_positions = {}
        for i, token in enumerate(sentence):
            if token[1][0][0] == self.punctuation_tag:
                punctuation_marks_positions[i] = token[0]  # stores the punctuation sign in a hash-table with their position as key for later retrieval
            else:
                sentence_without_punctuation_marks.append(token)
        return sentence_without_punctuation_marks, punctuation_marks_positions

    def get_sequence(self, pos_tags):
        """A method to disambiguate the part-of-speech tags attributed to the words using the context of the sentence."""
        tagged_tokens = pos_tags
        sentence_without_punctuation_marks, punctuation_marks_positions = self.__separate_punctuation_marks(tagged_tokens)
        error_in_desambiguation_process = False
        desambiguated_sentence = []
        token_counter = 1
        end_of_sentence = ("EOS", [("E", 1)])
        sentence_without_punctuation_marks.append(end_of_sentence)

        previous_states = [{"S": [1, "*"]}]
        for token in sentence_without_punctuation_marks:
            current_states = []
            dict_pos = {}
            token_counter += 1
            possible_tags = token[1]
            for tag in possible_tags:
                main_pos = tag[0].split("_")
                t = main_pos[0]
                dict_pos[t] = [0, ""]
                for key, value in previous_states[-1].items():
                    previous_state_prob = value[0]
                    emission_prob = tag[1]
                    u_v = "{}_{}".format(value[1], key)
                    transition_prob = self.data[t][u_v]
                    probability = previous_state_prob * emission_prob * transition_prob
                    if probability > dict_pos[t][0]:
                        dict_pos[t][0] = probability
                        dict_pos[t][1] = key
            previous_states.append(dict_pos)

        highest_score = "E"
        desambiguated_tag_list = []
        for i in previous_states[::-1]:
            try:
                highest_score = i[highest_score][1]
                desambiguated_tag_list.append(highest_score)
            except:
                error_in_desambiguation_process = True

        desambiguated_tag_list_reversed = desambiguated_tag_list[::-1][2:]
        desambiguated_sentence = []
        punctuation_marks_counter = 0
        for i, token in enumerate(tagged_tokens):
            if i in punctuation_marks_positions.keys():
                punct_mark_list = punctuation_marks_positions[i]
                pair = (punct_mark_list, self.punctuation_tag)
                desambiguated_sentence.append(pair)
                punctuation_marks_counter += 1
            else:
                desambiguated_pos = desambiguated_tag_list_reversed[i-punctuation_marks_counter]
                alternatives = tagged_tokens[i][1]
                for j, alternative in enumerate(alternatives):
                    base_tag = alternative[0].split("_")
                    if base_tag[0] == desambiguated_pos:
                        pair = (tagged_tokens[i][0], tagged_tokens[i][1][j][0])
                        desambiguated_sentence.append(pair)
                        break
        if error_in_desambiguation_process:
            return False
        return desambiguated_sentence


class Tokenizer:
    """A class to tokenize the text and separate it in sentences."""
    def __init__(self, filename=False):
        self.composed_tokens = {}

    def __separate_sentences(self, token_list):
        """A method to separate the tokenized text in sentences."""
        token_list = token_list
        end_of_sentence_markers = ["exclamation mark", "question mark", "period", "new line"]
        sentences = []
        current_sentence = []
        open_citation = False
        ignore_token = False
        if token_list[-1][1] not in end_of_sentence_markers:
            pair = ("\n", "new line")
            token_list.append(pair)
        for i, token in enumerate(token_list):
            if ignore_token:
                ignore_token = False
            else:
                if token[0] == '"':
                    open_citation = True
                current_sentence.append(token)
                if token[1] in end_of_sentence_markers:
                    try:
                        if token_list[i + 1][0] == '"' and open_citation:
                            current_sentence.append(token_list[i + 1])
                            open_citation = False
                            ignore_token = True
                    except:
                        pass
                    if len(current_sentence) > 1:
                        sentences.append(current_sentence)
                    current_sentence = []
        return sentences

    def tokenize(self, text):
        """Regex rules to extract the tokens from the text."""
        scanner = re.Scanner(
            [
                (r"\n", lambda scanner, token: (token, "new line")),
                (r'[„”"“”‘’‹›«»]', lambda scanner, token: (token, "quotation mark")),
                (r"(?:[a-zA-Z]\.){2,}", lambda scanner, token: (token, "acronym")),
                (r"[A-zA-ZÀ-ža-zà-ž’']+(?:-[A-zA-ZÀ-ža-zà-ž’']+)?", lambda scanner, token: (token, "word")),
                (r"(\d+(?:[\.,]\d+)?)+", lambda scanner, token: (token, "number")),
                (r"[0-9]+", lambda scanner, token: (token, "number")),
                (r"\.+(!?|\??)", lambda scanner, token: (token, "period")),
                (r",", lambda scanner, token: (token, "comma")),
                (r":", lambda scanner, token: (token, "colon")),
                (r";", lambda scanner, token: (token, "semicolon")),
                (r"[()]", lambda scanner, token: (token, "bracket")),
                (r"<>/+//-", lambda scanner, token: (token, "operator")),
                (r"\?+\.?", lambda scanner, token: (token, "question mark")),
                (r"!+\.?", lambda scanner, token: (token, "exclamation mark")),
                (r"[−/-—]", lambda scanner, token: (token, "hypen")),
                (r"[$€]", lambda scanner, token: (token, "symbol")),
                (r"[&\*•\|²]", lambda scanner, token: (token, "other")),
                (r"\s+", None),  # space // пробелы
                (r".", lambda scanner, token: (token, "notMatched")),  # ignore unmatched tokens // игнорировать нераспознанные токены
            ]
        )
        token_list = scanner.scan(text)  # word segmentation // выделение слов
        sentences = self.__separate_sentences(token_list[0])  # sentence segmentation // сегментация предложений
        return sentences
