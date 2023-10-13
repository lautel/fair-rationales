# Being Right for Whose Right Reasons?
This is the implementation of the framework described in the paper:
> Terne Sasha Thorn Jakobsen*, Laura Cabello*, Anders Søgaard [Being Right for Whose Right Reasons?](https://arxiv.org/abs/2306.00639). _In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Jul 2023._

We provide the code for reproducing our results, preprocessed data and fine-tuned models.

The data from our surveys can be found [here](https://github.com/terne/Being_Right_for_Whose_Right_Reasons).

## News

* 10-2023: Now the data is also available in [Huggingface](https://huggingface.co/datasets/coastalcph/fair-rationales)

## Repository Setup

You can clone this repository with submodules included issuing: <br>
`git clone git@github.com:lautel/fair-rationales.git`

1\. Create a fresh conda environment, and install all dependencies.
```text
conda create -n volta python=3.9.13
conda activate fair
pip install -r requirements.txt
```

2\. Install PyTorch (the following command will install torch version compatible with A100 cards)
```text
conda install pytorch=1.12.0=py3.9_cuda11.3_cudnn8.3.2_0 torchvision=0.13.0=py39_cu113 cudatoolkit=11.3 -c pytorch
```

3\. If you use a cluster, you may want to run commands like the following (if not loaded in `.bashrc` yet):
```text
module load anaconda3/5.3.1
module load cuda/11.4
```

## Repository structure

This repository contains the following folders:

* `data/` this is a placeholder. You should download the content from the [data repository](https://github.com/terne/Being_Right_for_Whose_Right_Reasons) and place it inside `data/`.
* `model_results/` contains the output files of the attribution methods for each model and population group (BO, BY, LO, LY, WO, WY). Please, refer to the paper for further details.
* `config/` contains a JSON file with the configuration of the project. Modify the data paths as needed. 
* `scripts/` contains the bash scripts to run the main Python scripts.
* `src/` contains the source code.

## License
This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@article{Jakobsen2023BeingRF,
  title={Being Right for Whose Right Reasons?},
  author={Terne Sasha Thorn Jakobsen and Laura Cabello and Anders S{\o}gaard},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.00639}
}
```
