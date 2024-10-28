import pickle
from tqdm import tqdm
from text_utils import preprocess_text


class Tokeniser:
    def __init__(self, lookup_pkl="utils/lookup_updated.pkl") -> None:
        with open(lookup_pkl, "rb") as f:
            word_to_id, id_to_word = pickle.load(f)
        self.words_to_ids = word_to_id
        self.ids_to_words = id_to_word
        self.default_token = 0
        self.preprocess = preprocess_text

    def tokenise_string(self, text: str, use_tqdm: bool = False) -> list[int]:
        split = self.preprocess(text)
        if use_tqdm:
            tokens = [
                (
                    self.words_to_ids[word]
                    if word in self.words_to_ids
                    else self.default_token
                )
                for word in tqdm(split)
            ]
        else:
            tokens = [
                (
                    self.words_to_ids[word]
                    if word in self.words_to_ids
                    else self.default_token
                )
                for word in split
            ]
        return tokens

    # def tokenise_list_of_words(self, text: str) -> list[int]:
    #     tokens = [
    #         self.word_to_id[word] if word in self.word_to_id else self.default_token
    #         for word in text
    #     ]
    #     return tokens

    # def _preprocess(self, text: str) -> list[str]:
    #     text = text.lower()
    #     text = text.replace(".", " <PERIOD> ")
    #     text = text.replace(",", " <COMMA> ")
    #     text = text.replace('"', " <QUOTATION_MARK> ")
    #     text = text.replace(";", " <SEMICOLON> ")
    #     text = text.replace("!", " <EXCLAMATION_MARK> ")
    #     text = text.replace("?", " <QUESTION_MARK> ")
    #     text = text.replace("(", " <LEFT_PAREN> ")
    #     text = text.replace(")", " <RIGHT_PAREN> ")
    #     text = text.replace("--", " <HYPHENS> ")
    #     text = text.replace("?", " <QUESTION_MARK> ")
    #     text = text.replace(":", " <COLON> ")
    #     return text.split()


if __name__ == "__main__":
    tokeniser = Tokeniser("utils/lookup.pkl")

    demo_text = (
        "Background. Metabolic acidosis is a clinical "
        "disturbance characterized by an increase in plasma acidity. "
        "Metabolic acidosis should be considered a sign of an "
        "underlying disease process. Identification of this underlying "
        "condition is essential to initiate appropriate therapy. "
        "Metabolic acidosis is a primary decrease in serum HCO 3. Rarely, "
        "metabolic acidosis can be part of a mixed or complex acid-base "
        "disturbance in which 2 or more separate metabolic or respiratory "
        "derangements occur together. In these instances, pH may not be "
        "reduced or the HCO 3 - concentration may not be low."
    )
    tokens = tokeniser.tokenise_string(demo_text)

    print(tokens)
