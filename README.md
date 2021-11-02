# A part-of-speech tagger for Sranan-Tongo based on a Trigram Hidden Markov Model. 

This project presents a prototype of a part-of-speech tagger for Sranan-Tongo using a 3-gram Hidden Markov Model. The model was trained on a set of 2945 sentences from three different sources:

1. Papers on Sranantongo" (Nickel Marilyn; Wilner John, Summer Institute of Linguistics. 1984) (208 sentences, 1571 tokens);
2. Sranan structure dataset (Winford Donald; Plag Ingo, The Atlas of Pidgin and Creole Language Structures Online. APiCS. 2013) (270 sentences, 2467 tokens);
3. Wortubuku fu Sranan Tongo. Online version. SIL International (2476 sentences, tokens 25597).

The sentences extracted from "Papers on Sranan Tongo" and "Sranan structure databases" are presumed to provide the training set with different grammatical constructions and to cover most of the morphosyntactic characteristics of the language, as they come from works that describe the language.
The sentences from the "Wortubuku fu Sranan Tongo" are taken from the examples in the dictionary entries that show how a headword is used in its various senses. Therefore, they are expected to be good candidates for adding lexical variation in the training set.

The sentences were manually annotated with the part-of-speech tags listed in the table below.


## Tag set

Tag|Description|Example
---|-----------|-------
AP|quantifier|moro, ala
AP|specifier|srefi, alamala
AT|article|a, den, wan
AUX|auxiliary/modal|musu,sa
CC|coordinating conjunction|èn, nanga
CS|subordinating conjunction|awinsi, te
COMPL|complementizer|taki, dati
COP|copula|a, na
DT|demostrative pronoun|dati, disi
DUT|particles from Dutch|dati, disi
EX|existential copula|de
FOC|focus marker|na
IN|preposition|te, abra
JJ|attributive adjective|bigi, redi
LOC|locational preposition|na
NEG|negation|no
NN|noun|boi, alen, oto
NP|proper noun|Sranan Tongo, Akuba
NUMB|numeral|twarfu, fosi, 
PRN|pronoun|mi, densrefi
PP$|possessive pronoun|mi, yu, en
RB|adverb|moro, agen
REL|relative pronoun|di, san, pe
ST|predicative adjective|faya, redi
TMA|time and aspect marker|ben, e, o
VB|verb|go, taki, sabi
UH|interjection|ei, we
WP|question word|san, suma


## The model

The trained model is divided into three files that contains statistical data obtained from the training set:

* *postag_distribution.json*: contains the frequency of the part-of-speech tags in the training set. It is used by the tagging algorithm to estimate the probabilies of the words not found in the training set;
* *emission_probabilities.json*: contains the probabilities of a word given a part-of-speech tag in the training set;
* *transition_probabilities.json*: contains the conditional probabilities of observing tag *ti* in a sentence, provided that before it appear tags *ti-2* and *ti-1*.

All of them are required by the part-of-speech tagging algorithm. The project's *data* folder already contains these files ready to use. They were generated on the 2945 sentences mentioned above. However, The model can be (re)trained on different datasets by running the training script *train.py*.

Use example
```
python train.py

```

## Training script

The training script (train.py) looks for training datasets in a specified folder ("/datasets" by default) and asks which one to use to train the model. The datasets must be in json format and contain the following basic structure:

```
{ "content": [
                {"parse": [
                    {"token": "word-form1", "postag": "part-of-speech tag1" },
                    {"token": "word-form2", "postag": "part-of-speech tag2" },
                    {...}
                    ]
                },
                {"parse": [
                    {"token": "word-form3", "postag": "part-of-speech tag3" },
                    {"token": "word-form4", "postag": "part-of-speech tag4" },
                    {...}
                    ]
                }
            ]
}
```

## Example code

The script below shows how to import and initialize the classes, define the general parameters and employ the methods to mark a text in Sranan-Tongo with part-of-speech tags.

```
# import classes
import os
from tagger import Tokenizer, Emission, Transition

# example text
sample_text = 'Kofi lobi a umapikin.'

# get working folder
root_folder = os.getcwd()

# path to the folder with required json files
data_folder = root_folder + "/data/"
postag_dist = data_folder + "postag_distribution.json"
emission_prob = data_folder + "emission_probabilities.json"
transition_prob = data_folder + "transition_probabilities.json"

# object initialization
tokenizer = Tokenizer()
emission = Emission(postag_dist, emission_prob) # requires the json files with the POS tags distribution and the emission probabilities as parameters
transition = Transition(transition_prob) # requires the json file with the transition probabilities as parameter

# tokenization proceses (separates the text into a list of sentences and tokens )
tokenized_sentences = tokenizer.tokenize(sample_text) 
for sentence in tokenized_sentences:
    # process for marking the text with part-of-speech tags
    tagged_tokens = emission.get_emission_probabilities(sentence)  # assigns the possible tags to a word form
    disambiguated_sequence = transition.get_sequence(tagged_tokens)  # disambiguates the assigned tags in the context of the sentence
    print(disambiguated_sequence)
```


## Testing the model

The trained model can be tested using the script *test.py* and passing a specified dataset as argument. This process with generate a confusion matrix for the part-of-speech tags with their respective precision, recall and f1-score values.

Use example:

```
python test.py testing/test_text1.json

```


