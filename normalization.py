from unidecode import unidecode
import pandas as pd
import re
import os

class Normalizer():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Loads all known stopwords from file
    NORMALIZATION_STOPWORDS = set()
    for language in ["brazilian", "english", "spanish"]:
        sws = pd.read_csv(os.path.join(ROOT_DIR, language + "_stopwords.csv")).applymap(lambda x: x.strip()).word.unique()
        for sw in sws:
            NORMALIZATION_STOPWORDS.add(sw)

    # Symbols for normalize_string function
    symbols = {
        "[", "]", "{", "}", "(",
        ")", "_", "-", "+", "=",
        "/", "*",  "º", "ª", ",",
        ".", ":", ";", "?", "!",
        "'", "\"", "%", "$", "#"
    }

    def normalize(self, string):
        string = unidecode(string)
        new_string = ""
        for c in string:
            if c not in self.symbols:
                new_string += c
            else:
                new_string += " "
        return self.remove_tags(" ".join(new_string.lower().replace("\t", " ").split()).strip())

    def remove_tags(self, string):
        n_string = re.sub("<[^<]*>", "", string)
        n_string = re.sub("&[^; ]*;", " ", n_string)
        return n_string

    def remove_stopwords(self, string):
        new_string = ""
        for word in str(string).split():
            if word not in self.NORMALIZATION_STOPWORDS:
                new_string += word + " "

        return new_string.strip()