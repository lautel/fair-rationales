import ast
import json
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from data.dataset_loader import SupportedDatasets
from typing import Dict, Any


def main(method: str, dataset: str, modelname: str) -> None:
    """Compute main results for model attribution scores:
    binarize explanations to get model rationales comparable to human's rationales;
    compute the topkd tokens.

    Results are stored in a CSV file.

    Args:
        method (str): explainability method
        dataset (str): name of the dataset
        modelname (str): name of the model (following nomenclature from huggigface)
    """

    # Load configuration
    config = json.load(open("config/config.json", "r"))

    # Create output directory
    modelname = modelname.replace("/", "_")
    output_dir = os.path.join(config["model_importance"], method)
    os.makedirs(output_dir, exist_ok=True)

    # Load data from the annotators
    assert dataset in [
        SupportedDatasets.SST2,
        SupportedDatasets.DYNASENT,
        SupportedDatasets.COSE,
    ]
    glob_path = glob(f"{config[dataset+'_annotations']['root']}/**_processed.csv")

    for path in glob_path:
        group = os.path.basename(os.path.normpath(path))[:2]
        print()
        print(path, group)
        df_group = pd.read_csv(path, converters={"rationale_binary": ast.literal_eval})
        df_group.dropna(inplace=True)
        df_group.sort_index(inplace=True)

        # Load data from the model
        if method == "attn_rollout":
            method_attr = "attention"
        else:
            method_attr = "lrp"
        try:
            df = pd.read_csv(
                f"{output_dir}/{method}_{modelname}_{dataset}_attention.csv",
                converters={
                    method_attr: ast.literal_eval,
                    "tokens_merged": ast.literal_eval,
                },
            )
        except ValueError:
            df = pd.read_csv(
                f"{output_dir}/{method}_{modelname}_{dataset}_attention.csv",
                converters={"tokens_merged": ast.literal_eval},
            )
        df.sort_index(inplace=True)
        df["rationale_binary"] = ""

        ###################
        ### COMPUTE RLR ###
        ###################
        # RLR represents the ratio of rationale length to its input length
        # Note this approach is generalistic. It could also be done by group and not an overall average
        rlr_all = []
        rationale_length = []
        for x in df_group["rationale_binary"]:
            rlr_all.append(sum(x) / len(x))
            rationale_length.append(sum(x))
        rlr = np.mean(rlr_all)
        print("Average RLR {:.2f}% ({})".format(100 * rlr, group))
        print("Average rationale length {:.2f}".format(np.mean(rationale_length)))

        #######################################
        ### ADD RATIONALE BINARY FOR MODELS ###
        #######################################
        # https://arxiv.org/pdf/2205.11097.pdf
        sentences_unequal_length = []
        matching = 0

        id_name = "originaldata_id"
        if dataset == SupportedDatasets.SST2:
            data_id_all = df_group["sst2_id"].astype(int)
        else:
            data_id_all = df_group[id_name]

        for data_id in data_id_all:
            if (
                dataset == SupportedDatasets.SST2
                or dataset == SupportedDatasets.DYNASENT
            ):
                try:
                    xi = df.loc[df[id_name] == data_id, method_attr].values[0]
                except KeyError:
                    id_name = "sst2_id"
                    xi = df.loc[df[id_name] == data_id, method_attr].values[0]
            else:
                # variants = ["_A", "_B", "_C", "_D", "_E"]
                letter = df_group.loc[
                    df_group[id_name] == data_id, "original_label"
                ].values[0]
                xi = df.loc[
                    df[id_name] == (data_id + "_" + letter), method_attr
                ].values[0]
            if isinstance(xi, str):
                xi = [float(ii) for ii in xi[1:-1].split(",")]

            yi = df_group.loc[data_id_all == data_id, "rationale_binary"].values[0]
            sentence = df_group.loc[data_id_all == data_id, "sentence"].values[0]

            if dataset == SupportedDatasets.COSE:
                # Correct specific cases:
                if sentence.startswith("I 'm"):
                    sentence = "I'm" + sentence[len("I 'm") :]
                    yi[0] = max(yi[0], yi[1])
                    yi.pop(1)
                if " - " in sentence:
                    idx_dash = sentence.index(" - ")
                    idx_dash_token = sentence.split().index("-")
                    sentence = sentence[:idx_dash] + "-" + sentence[idx_dash + 3 :]
                    yi[idx_dash_token - 1] = max(
                        yi[idx_dash_token - 1], yi[idx_dash_token + 1]
                    )
                    yi.pop(idx_dash_token)
                    yi.pop(idx_dash_token)
                if sentence.endswith("? ."):
                    sentence = sentence[:-2] + "."
                    yi[-2] = max(yi[-2], yi[-1])
                    yi = yi[:-1]
                # Remove the model's attention to the concatenated answer (last 1 or 2 tokens)
                xi = xi[: -(len(xi) - len(sentence.split()))]

            if (
                len(yi) - len(xi) == 1
                and sentence.startswith("...")
                and df.loc[df[id_name] == data_id, "tokens_merged"].values[0][0]
                != "..."
            ):
                df.at[df.loc[df[id_name] == data_id].index[0], "tokens_merged"] = [
                    "..."
                ] + df.loc[df[id_name] == data_id, "tokens_merged"].values[0]
                xi = [0] + xi
                df.at[df.loc[df[id_name] == data_id].index[0], method_attr] = xi
            if len(xi) == len(yi):
                top_kd = int(np.round(rlr * len(xi)))
                top_kd_tokens = np.argsort(-np.asarray(xi))[:top_kd]
                xi_binary = np.zeros((1, len(xi)), dtype=int)
                xi_binary[:, top_kd_tokens] = 1
                if (
                    dataset == SupportedDatasets.SST2
                    or dataset == SupportedDatasets.DYNASENT
                ):
                    dfidx = df.loc[df[id_name] == data_id].index[0]
                else:
                    dfidx = df.loc[df[id_name] == data_id + "_" + letter].index[0]

                df.at[dfidx, "rationale_binary"] = xi_binary.tolist()[0]
                matching += 1
            else:
                if (
                    dataset == SupportedDatasets.SST2
                    or dataset == SupportedDatasets.DYNASENT
                ):
                    sentences_unequal_length.append(
                        (
                            df_group.loc[data_id_all == data_id].index[0],
                            data_id,
                            df.loc[df[id_name] == data_id, "tokens_merged"].values[0],
                            df_group.loc[data_id_all == data_id, "sentence"]
                            .values[0]
                            .split(),
                        )
                    )
                else:
                    sentences_unequal_length.append(
                        (
                            df_group.loc[data_id_all == data_id].index[0],
                            data_id,
                            df.loc[
                                df[id_name] == data_id + "_" + letter, "tokens_merged"
                            ].values[0],
                            df_group.loc[data_id_all == data_id, "sentence"]
                            .values[0]
                            .split(),
                        )
                    )
                # print(data_id)
                # print(len(xi), df.loc[df[id_name] == data_id, "tokens_input"].values[0])
                # print(len(xi), df.loc[df[id_name] == data_id, "tokens_merged"].values[0])
                # print(len(yi), df_group.loc[df_group[id_name] == data_id, "sentence"].values[0].split())

        print(f"{matching} sentences added. {len(sentences_unequal_length)} discarded.")
        df.to_csv(
            f"{output_dir}/{method}_{modelname}_{dataset}_attention_topkd_{group}.csv",
            index=True,
        )
        print(
            f"{output_dir}/{method}_{modelname}_{dataset}_attention_topkd_{group}.csv saved!"
        )
        with open(f"{output_dir}/sentences_unequal_length_{dataset}.txt", "w") as f:
            for item in sentences_unequal_length:
                f.write("\nidx " + str(item[0]))
                f.write(f"\n{id_name} " + str(item[1]))
                f.write("\nmodel " + str(item[2]))
                f.write("\nannot " + str(item[3]) + "\n")
        print(f"{output_dir}/sentences_unequal_length_{dataset}.txt saved!")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
