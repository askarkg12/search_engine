import pickle
import datasets
from collections import Counter
from tqdm import tqdm
from text_utils import preprocess_text_threshold, NEW_COUNT_THRESHOLD


words_to_ids: dict[str, int] = {}
ids_to_words: dict[int, str] = {}

if __name__ == "__main__":
    lookup_pkl_path = "utils/lookup.pkl"
    with open(lookup_pkl_path, "rb") as f:
        words_to_ids, ids_to_words = pickle.load(f)

    dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")

    splits = ["train", "test", "validation"]
    text_corpus_list = []

    for split in splits:

        queries = dataset[split]["query"]
        text_corpus_list.append(" ".join(queries))

        passages = dataset[split]["passages"]
        for passage in tqdm(passages, desc=split + " passages"):
            x = " ".join(passage["passage_text"])
            text_corpus_list.append(" ".join(passage["passage_text"]))

    text_corpus = " ".join(text_corpus_list)
    # pass
    print("Processing text")
    words = preprocess_text_threshold(text_corpus)

    print("Counting words")
    word_counts = Counter(words)

    limited_word_counts = set(
        [word for word in word_counts if word_counts[word] > NEW_COUNT_THRESHOLD]
    )

    old_vocab = set(words_to_ids)

    diff_vocab = limited_word_counts.difference(old_vocab)

    new_word_to_id = {
        word: id for id, word in enumerate(diff_vocab, start=len(old_vocab))
    }
    new_id_to_word = {id: word for word, id in new_word_to_id.items()}

    words_to_ids.update(new_word_to_id)
    ids_to_words.update(new_id_to_word)

    dataset = (words_to_ids, ids_to_words)

    with open("utils/lookup_updated.pkl", "wb") as f:
        pickle.dump(dataset, f)
