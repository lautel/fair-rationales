# Utils for plotting
from typing import List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

rc = {
    "font.size": 11,
    "axes.labelsize": 15,
    "legend.fontsize": 10.0,
    "axes.titlesize": 32,
    "xtick.labelsize": 20,
    "ytick.labelsize": 16,
}
plt.rcParams.update(**rc)
matplotlib.rcParams["axes.linewidth"] = 0.5  # set the value globally


def _get_canvas(
    words: List[str], x: List[float], H: int = 200, W: int = 50
) -> Tuple[np.array, List[float]]:
    ntoks = len("".join(words))
    W_all = W * ntoks
    fracs = [len(w_) / ntoks for w_ in words]
    delta_even = int(W_all / ntoks)

    X = np.zeros((H, W_all))
    x0 = 0

    x_centers = []
    for i, (w_, b) in enumerate(zip(words, x)):
        delta = int((len(w_) / ntoks) * W_all)
        delta = int((0.85 * delta_even + 0.15 * delta))
        X[:, x0 : x0 + delta] = b

        x_centers.append(x0 + int(delta / 2))
        x0 = x0 + delta
    X = X[:, :x0]
    return X, x_centers


def plot_sentence(words: List[str], x: List[float], H0: int = 100) -> None:
    """Visualisation of the attribution scores per word in a sentence.

    :param words: Sentences to visualize given as a list of words.
    :param x: Attribution scores by word, given by an explainability method.
    :param H0: Height of the canvas
    :return: None
    """
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    f, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(10, 2), gridspec_kw={"height_ratios": [1]}
    )

    x_, x_centers = _get_canvas(words, x, H=H0, W=52)
    h = ax.imshow(x_, cmap="seismic", vmin=-1, vmax=1.0, alpha=0.75)

    for k, word in zip(x_centers, words):
        ax.text(k, H0 / 2, word, ha="center", va="center")
    plt.colorbar(h)
    plt.axis("off")
    plt.show()


def plot_group_sentences(
    words_list: List[List[str]],
    x_list: List[List[float]],
    H0: int = 100,
    out_name: str = "sentence_",
) -> None:
    """Visualisation of the attribution scores per word in a sentence.

    :param words_list: List of sentences to visualize. Each sentence is a list of words.
    :param x_list: List of lists of attribution scores, one per sentence and explainability method.
    :param H0: Height of the canvas
    :param out_name: Output file name
    :return: None
    """
    assert len(words_list) == len(x_list)
    f, ax = plt.subplots(
        nrows=len(words_list),
        ncols=1,
        figsize=(10, 1.5),
        # gridspec_kw={'height_ratios': [1]*len(words_list)}
    )

    i = 0
    for words, x in zip(words_list, x_list):
        x_, x_centers = _get_canvas(words, x, H=H0, W=252)
        for k, word in zip(x_centers, words):
            ax[i].text(k, H0 / 2, word, ha="center", va="center", size="xx-large")
        ax[i].set_xticks([])  # remove numbers x
        ax[i].set_yticks([])  # remove numbers y
        # ax[i].axis('off')
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        if i % 2 == 0:
            ax[i].set_ylabel("AR", rotation=90)
            h = ax[i].imshow(x_, cmap="bwr", vmin=0, vmax=1.0, alpha=0.75)
        else:
            ax[i].set_ylabel("LRP", rotation=90)
            h = ax[i].imshow(x_, cmap="bwr", vmin=-1, vmax=1.0, alpha=0.75)
        i += 1

    cb = plt.colorbar(h)
    cb.remove()  # remove legend
    f.tight_layout()
    # plt.subplots_adjust(wspace=0.0, hspace=0.01)
    # plt.show()
    plt.savefig(f"{out_name}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import numpy as np

    # Example 1 from dynasent, common_dyna.loc[common_dyna['originaldata_id'] == 'r2-0019963']
    sentence1 = "We were helped by two awesome ladies."
    words1 = sentence1.split()
    bo = [0, 0, 1, 0, 0, 1, 0]
    by = [0, 0, 0, 0, 0, 1, 1]
    lo = [0, 0, 0, 0, 0, 1, 0]
    ly = [0, 0, 0, 0, 0, 1, 0]
    wo = [0, 0, 1, 0, 1, 0, 1]
    wy = [0, 0, 1, 0, 0, 1, 1]

    # LRP values
    x_lrp_roberta1 = [
        -0.04165829345583916,
        0.011820955201983452,
        0.09553434699773788,
        0.046529725193977356,
        -0.03201446682214737,
        0.12419025599956512,
        -0.09478265792131424,
    ]
    x_lrp_roberta1 = np.asarray(x_lrp_roberta1) / max(x_lrp_roberta1)
    print(np.sum(x_lrp_roberta1), x_lrp_roberta1)
    assert len(words1) == len(x_lrp_roberta1)
    # plot_sentence(words1, x_lrp_roberta1)
    # Attention Rollout values
    x_attn_roberta1 = [
        5.441777508192762,
        7.241374219200693,
        7.214393990569738,
        6.078160835664312,
        4.209768514161674,
        4.255383596708715,
        4.910166522262202,
    ]
    x_attn_roberta1 = np.asarray(x_attn_roberta1) / max(x_attn_roberta1)
    print(np.sum(x_attn_roberta1), x_attn_roberta1)
    assert len(words1) == len(x_attn_roberta1)
    # plot_sentence(words1, x_attn_roberta1)

    plot_group_sentences(
        [words1, words1], [x_attn_roberta1, x_lrp_roberta1], "utils/dyna_sentence_1"
    )

    # Example 2, r2-0019567
    sentence2 = "The waiters handled the stress very well."
    words2 = sentence2.split()

    # LRP values
    x_lrp_roberta2 = [
        0.04965570569038391,
        -0.11558359861373901,
        -0.0093940244987607,
        0.025117048993706703,
        0.09740941226482391,
        -0.15023285150527954,
        0.1506640762090683,
    ]
    x_lrp_roberta2 = np.asarray(x_lrp_roberta2) / max(x_lrp_roberta2)
    print(np.sum(x_lrp_roberta2), x_lrp_roberta2)
    assert len(words2) == len(x_lrp_roberta2)
    # plot_sentence(words2, x_lrp_roberta2)
    # Attention Rollout values
    x_attn_roberta2 = [
        5.623765227653896,
        7.079231631865372,
        6.279358412504709,
        5.435459153092983,
        6.0899405864577405,
        4.839571933835503,
        4.6595628618416445,
    ]
    x_attn_roberta2 = np.asarray(x_attn_roberta2) / max(x_attn_roberta2)
    print(np.sum(x_attn_roberta2), x_attn_roberta2)
    assert len(words2) == len(x_attn_roberta2)
    # plot_sentence(words2, x_attn_roberta2)

    plot_group_sentences(
        [words2, words2], [x_attn_roberta2, x_lrp_roberta2], "utils/dyna_sentence_2"
    )
