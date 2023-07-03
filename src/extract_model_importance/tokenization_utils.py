# Code adapted from https://github.com/beinborn/relative_importance/blob/main/extract_model_importance/tokenization_util.py

# Aligning the tokenization of different language model tokenizers
# with the tokenization in the eye-tracking corpora is really tricky.
# We did our best to account for as many cases as possible.
# Some cases are so specific that they would need to be hard-coded.
# For example, the ZUCO corpus contains a few instances of "U.S" which
# is seen as a single token but separated by most tokenizers.
# We decided to simply ignore these very specific cases but encourage you to do better.
import re
from typing import List, Tuple


def merge_subwords(
    tokens: List[str], summed_importance: List[float]
) -> Tuple[List[str], List[float]]:
    adjusted_tokens = []
    adjusted_importance = []

    current_token = ""
    current_importance = 0

    # Tokenizers use different word piece separators. We simply check for both here
    word_piece_separators = ("##", "_")
    for i, token in enumerate(tokens):
        # We sum the importance of word pieces
        current_importance += summed_importance[i]

        # Identify word piece
        if token.startswith(word_piece_separators):
            # skip the hash tags
            current_token += token[2:]

        else:
            current_token += token

        # Is this the last token of the sentence?
        if i == len(tokens) - 1:
            adjusted_tokens.append(current_token)
            adjusted_importance.append(current_importance)

        else:
            # Are we at the end of a word?
            if not tokens[i + 1].startswith(word_piece_separators):
                # append merged token and importance
                adjusted_tokens.append(current_token)
                adjusted_importance.append(current_importance)

                # reset
                current_token = ""
                current_importance = 0
    return adjusted_tokens, adjusted_importance


# Word piece tokenization splits words separated by hyphens. Most eye-tracking corpora don't do this.
# This method sums the importance for tokens separated by hyphens.
def merge_hyphens(
    tokens: List[str], importance: List[float]
) -> Tuple[List[str], List[float]]:
    adjusted_tokens = []
    adjusted_importance = []

    if "-" in tokens:
        # Get all indices of -
        indices = [i for i, x in enumerate(tokens) if x == "-"]
        i = 0
        while i < len(tokens):
            if i + 1 in indices and i + 2 < len(tokens):
                combined_token = tokens[i] + tokens[i + 1] + tokens[i + 2]
                combined_heat = importance[i] + importance[i + 1] + importance[i + 2]
                i += 3
                adjusted_tokens.append(combined_token)
                adjusted_importance.append(combined_heat)
            else:
                adjusted_tokens.append(tokens[i])
                adjusted_importance.append(importance[i])
                i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


# Word piece tokenization splits parentheses and currency symbols as separate tokens. This is not done in Zuco.
def merge_symbols(
    tokens: List[str], importance: List[float]
) -> Tuple[List[str], List[float]]:
    initial_symbols = ["(", "$", "€", '"', "'", "`"] + ["`", "``", "''"]
    end_symbols = [")", "%", '"', "'"] + [";", "''", "'d", "'ll", ":", "'n", ".", "'."]
    all_symbols = initial_symbols + end_symbols
    # First check if anything needs to be done
    if any(token in all_symbols for token in tokens):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens) - 1:
            combined_token = tokens[i]
            combined_heat = importance[i]

            # Nothing to be done for the last token
            if i <= len(tokens) - 2:
                # Glue the parentheses back to the token
                if tokens[i] in initial_symbols:
                    combined_token = combined_token + tokens[i + 1]
                    combined_heat = combined_heat + importance[i + 1]
                    i += 1

                if i < len(tokens) - 1 and tokens[i + 1] in end_symbols:
                    combined_token = combined_token + tokens[i + 1]
                    combined_heat = combined_heat + importance[i + 1]
                    i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


def merge_albert_tokens(
    tokens: List[str], importance: List[float]
) -> Tuple[List[str], List[float]]:
    adjusted_tokens = []
    adjusted_importance = []
    i = 0
    # We ignore the last token [SEP]
    while i < len(tokens) - 1:
        combined_token = tokens[i]
        combined_heat = importance[i]
        # Nothing to be done for the last token
        if i < (len(tokens) - 2):
            while not tokens[i + 1].startswith("▁"):
                combined_token = combined_token + tokens[i + 1]
                combined_heat = combined_heat + importance[i + 1]
                i += 1
                if i == len(tokens) - 2:
                    break
        combined_token.replace("▁", "")
        adjusted_tokens.append(combined_token)
        adjusted_importance.append(combined_heat)
        i += 1
    # Add the last token
    adjusted_tokens.append(tokens[i])
    adjusted_importance.append(importance[i])
    return adjusted_tokens, adjusted_importance


def _get_token(st: List[str], i: int) -> int:
    word = st[:i].split(" ")[-1] + st[i:].split(" ")[0]  # else return the word
    word_list = st[: i + len(word)].split()
    return len(word_list)


def _select_token_from_list(token_ids: List[int], sentence_half: str) -> int:
    if len(token_ids) == 1:
        return token_ids[0]
    else:
        # get number closest to a given value
        return min(token_ids, key=lambda x: abs(x - len(sentence_half.split())))


def merge_cose_whitespaces_sentence(sentence: str) -> Tuple[str, List[int], int]:
    dash = False
    rationale_id_merged = []
    word_to_list_of_word_id = {
        word: [
            _get_token(sentence, w.start())
            for w in re.finditer(r"\b" + re.escape(word) + r"\b", sentence)
        ]
        for word in sentence.split()
    }
    for k, v in word_to_list_of_word_id.items():
        if len(v) == 0:
            word_to_list_of_word_id[k] = [
                _get_token(sentence, w.start())
                for w in re.finditer(re.escape(k), sentence)
            ]

    matches = reversed(list(re.finditer(r'(\w+)\s([?,.!"](?:|$))', sentence)))
    for m in matches:
        span = m.span()
        words = m.group().split()
        try:
            rationale_id_merged.append(
                _select_token_from_list(
                    word_to_list_of_word_id[words[1]], sentence[: span[1]]
                )
            )
        except KeyError:
            print(sentence, words)
        sentence = (
            sentence[: span[0]]
            + sentence[span[0] : span[1]].replace(" ", "")
            + sentence[span[1] :]
        )

    matches_contractions = reversed(
        list(
            re.finditer(r"(\w+)\s(\'s|\'re|n\'t|\'ll|\'ve|\'t|\'am|\'m|\'d)", sentence)
        )
    )
    for m in matches_contractions:
        span = m.span()
        words = m.group().split()
        rationale_id_merged.append(
            _select_token_from_list(
                word_to_list_of_word_id[words[1]], sentence[: span[1]]
            )
        )
        sentence = (
            sentence[: span[0]]
            + sentence[span[0] : span[1]].replace(" ", "")
            + sentence[span[1] :]
        )

    # Correct specific cases:
    if " - " in sentence:
        idx_dash = sentence.index(" - ")
        dash = sentence.split().index("-")
        sentence = sentence[:idx_dash] + "-" + sentence[idx_dash + 3 :]
        rationale_id_merged.append(
            _select_token_from_list(
                word_to_list_of_word_id["-"], sentence[: idx_dash + 3]
            )
        )
        # Also the word to the left and right side of the dash
        rationale_id_merged.append(rationale_id_merged[-1] + 1)
    if sentence.endswith("? ."):
        sentence = sentence[:-2] + "."
        rationale_id_merged.append(max(word_to_list_of_word_id["."]))

    return sentence, rationale_id_merged, dash


def merge_cose_whitespaces(
    ii: int, sentence: str, rationale_ids: List[int], rationales_str: str
) -> Tuple[str, str, str]:
    # print(ii, sentence, rationale_ids, rationales_str)
    # contractions = ["'s", "'re", "n't", "'ll", "'ve", "'t", "'am", "'d"]
    # doc = re.sub(r'\s([?,.!"](?:\s|$))', r'\1', sentence)  # remove spaces before ? , . !
    # # remove spaces before contractions.
    # If the apostrophe ' has a different format for some reason, then it won't work.
    # for contract in contractions:
    #     doc = doc.replace(" " + contract, contract)
    # doc = doc.replace("[sep]", sep_token)

    rationale_ids = rationale_ids.split(",")
    if len(rationale_ids) == 1 and rationale_ids[0].endswith(".0"):
        rationale_ids[0] = rationale_ids[0][: -len(".0")]

    # Word mapping accounting for a word appearing more than once in the sentence
    sentence, rationale_id_merged, dash = merge_cose_whitespaces_sentence(sentence)
    # if the merged token was selected as rationale, remove it
    rationale_id_merged = list(set(rationale_id_merged))
    idxes = [
        rationale_ids.index(str(ii))
        for ii in rationale_id_merged
        if str(ii) in rationale_ids
    ]
    for idx in sorted(idxes, reverse=True):
        str_ = rationales_str.split(",")[idx]
        if not str_:
            str_ = ","
        if idx > 1:
            rationales_str = (
                ",".join(rationales_str.split(",")[: idx - 1])
                + ","
                + re.sub(
                    r"(\w+)?,(" + re.escape(str_) + ")",
                    r"\1\2",
                    ",".join(rationales_str.split(",")[idx - 1 :]),
                    1,
                )
            )
        else:
            rationales_str = re.sub(
                r"(\w+)?,(" + re.escape(str_) + ")",
                r"\1\2",
                ",".join(rationales_str.split(",")),
                1,
            )
    # Clean commas selection in rationales
    if rationales_str.startswith(","):
        rationales_str = rationales_str[1:]
    if rationales_str.endswith(","):
        rationales_str = rationales_str[:-1]

    rationale_dic = {
        int(i): 0 for i in rationale_ids if int(i) not in rationale_id_merged
    }
    # rationale_ids = [int(i) for i in rationale_ids
    #                  if int(i) not in rationale_id_merged]
    for idm in sorted(rationale_id_merged, reverse=False):
        # rationale_ids = [i-1 if i >= idm else i for i in rationale_ids]
        for i in rationale_dic.keys():
            if int(i) >= idm:
                rationale_dic[i] += 1
    final_rationale_ids = []
    for k, v in rationale_dic.items():
        final_rationale_ids.append(int(k - v))
    try:
        # Clean up any possible ",," in the string
        while ",," in rationales_str:
            rationales_str = rationales_str.replace(",,", ",")
        if rationales_str.startswith(".,"):
            rationales_str = rationales_str[2:]
        if dash and len(final_rationale_ids) != len(rationales_str.split(",")):
            final_rationale_ids.append(dash)
        assert len(final_rationale_ids) == len(rationales_str.split(","))
    except AssertionError:
        print("AssertionError:", final_rationale_ids, rationales_str)
        final_rationale_ids = []
        rationales_str = ""
    # print(ii, final_rationale_ids, rationales_str)
    return sentence, str(final_rationale_ids)[1:-1], rationales_str


# These are just some tests for the methods
# print(merge_hyphens(["this", "co", "-", "exists", "peaceful", "##ly", "today"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
# print(merge_symbols(["Check", "this", "(", "February", ",", "1985", ")"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
# print(merge_symbols(["Check", "this", "(", "1985", ")"], [0.1, 0.2, 0.3, 0.4, 0.5]))
# print(merge_symbols(["Check", "this", "(", ")", "okay"], [0.1, 0.2, 0.3, 0.4, 0.5]))
# print(merge_symbols(["It", "costs", "$", "200", "."], [0.1, 0.2, 0.3, 0.4, 0.5]))

# Check merging of subwords
# print(merge_albert_tokens(['[CLR]', '▁presents', '▁a', '▁good', '▁case', '▁while', '▁failing', '▁to', '▁provide', '▁a', '▁reason', '▁for', '▁us', '▁to', '▁care', '▁beyond', '▁the', '▁very', '▁basic', '▁', 'dict', 'um', 's', '▁of', '▁human', '▁dec', 'ency', '[SEP]'], [0.21363762069336079, 0.06875212381891577, 0.014697464273896824, 0.02376395684040837, 0.05140783800073661, 0.027206018198431502, 0.020411475649548733, 0.012897350417446742, 0.01881216717103111, 0.012974560274041994, 0.0335241384655605, 0.01461099640793805, 0.018708945235305297, 0.014533449537614894, 0.041455039447895116, 0.023473885808694913, 0.012837706137504347, 0.020037277734046212, 0.02037590565738258, 0.0164441674593348, 0.098617052940637, 0.03552736363748528, 0.016664912663178762, 0.0145626107970789, 0.022517101256231506, 0.044144579179509834, 0.031474810989978494, 0.05592948130680448]))

# # Check what happens if no merging
# print(merge_albert_tokens(['[CLR]', '▁presents', '▁a', '▁good', '▁case','[SEP]' ], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
