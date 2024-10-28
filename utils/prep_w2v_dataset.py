import collections
import more_itertools
import pickle
import numpy as np
from tqdm import tqdm
from text_utils import preprocess_text_threshold
import datasets
import pathlib

from tokeniser import Tokeniser

tokeniser = Tokeniser("utils/lookup_updated.pkl")


dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")

splits = ["train", "test", "validation"]
passages_corpus_list = []
all_queries = []

for split in splits:

    queries = dataset[split]["query"]

    # TODO This is a better option
    # all_queries.extend(queries)

    # No isolation of queries here
    passages_corpus_list.append(" ".join(queries))

    passages = dataset[split]["passages"]
    for passage in tqdm(passages, desc=split + " passages"):
        x = " ".join(passage["passage_text"])
        passages_corpus_list.append(" ".join(passage["passage_text"]))

text_corpus = " ".join(passages_corpus_list)
# pass

passages_tokens_full = tokeniser.tokenise_string(text_corpus)

# Hopefully saves memory
del text_corpus

# queries_tkns = [tokeniser.tokenise(query) for query in tqdm(all_queries)]

passages_token_counts = collections.Counter(passages_tokens_full)
passages_tokens_len = len(passages_tokens_full)


# Subsample corpus
def subsample_corpus(corpus_tokens, threshold, tokens_len):
    token_counts = collections.Counter(corpus_tokens)

    def keep_probability(token) -> float:
        return 1 - np.sqrt((tokens_len * threshold) / token_counts[token])

    tqdm_tkn = tqdm(corpus_tokens, desc="Subsampling")

    subsampled = [
        token for token in tqdm_tkn if keep_probability(token) > np.random.ranf()
    ]
    return subsampled


# print("Saubsampling...")
passages_tokens_subsampled = subsample_corpus(
    corpus_tokens=passages_tokens_full, threshold=1e-5, tokens_len=passages_tokens_len
)
# print("Finished subsampling")

context_window = 5
if context_window % 2 == 0:
    raise Exception(
        f"Context Window must be an odd number, currently: {context_window}"
    )
center_i = context_window // 2

windows = more_itertools.windowed(passages_tokens_subsampled, context_window)

wind_tq = tqdm(windows, desc="Sliding window", total=len(passages_tokens_subsampled))
neg_sample_count = 20

inputs = []
targets = []
neg_samples = []
for tkn_wind in wind_tq:
    inputs.append(tkn_wind[center_i])
    targets.append(tkn_wind[:center_i] + tkn_wind[center_i + 1 :])
    # TODO
    negs = [
        passages_tokens_full[id]
        for id in np.random.randint(0, passages_tokens_len, neg_sample_count)
    ]
    neg_samples.append(negs)

pkl_dataset = (inputs, targets, neg_samples)

dataset_dir = pathlib.Path("dataset/ms_marco")
dataset_dir.mkdir(exist_ok=True, parents=True)
with open(dataset_dir / "w2v_finetune.pkl", "wb") as f:
    pickle.dump(pkl_dataset, f)
