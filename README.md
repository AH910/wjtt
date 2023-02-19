# wjtt
Implementation of Weighted Just Train Twice (MSc Thesis).

## Abstract
In some cases, standard machine learning models do well on average, but fail on certain groups of the data. This can happen when the data consists of several heterogeneous groups. For this situation, we propose the method Weighted Just Train Twice, a generalization of JTT. Our implementation is based on the code of [Just Train Twice](https://arxiv.org/pdf/2107.09044.pdf) and [group DRO](https://arxiv.org/abs/1911.08731).

## Packages
The packages needed for our code can be found in `requirements.txt`. They can be installed with the command `pip install -r requirements.txt`.

## Datasets
- **Waterbirds:** Our code expects the waterbird dataset in the directory `wjtt/data/waterbird_complete95_forest2water2`. The folder `waterbird_complete95_forest2water2` can be downloaded by clicking [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

- **CelebA:** Our code expects the celebA dataset in the directory `wjtt/data/celebA`, where the folder `celebA` should contain `img_align_celeba`, `list_attr_celeba.csv` and `list_eval_partition.csv`. All of this can be downloaded from  [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

- **CivilComments:** The CivilComments dataset can be downloaded [here](https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/). Our code expects the dataset at `wjtt/data/CivilComments/all_data_with_identities.csv`.

- **MultiNLI:** The MultiNLI dataset can be downloaded [here](https://github.com/kohpangwei/group_DRO#multinli-with-annotated-negations). In the directory `wjtt/data/multiNLI/` our code expects
    - metadata_preset.csv
    - cached_dev_bert-base-uncased_128_mnli
    - cached_dev_bert-base-uncased_128_mnli-mm
    - cached_train_bert-base-uncased_128_mnli


## Run the method
The method can be run using the command `python wjtt.py --dataset ... --num_epochs ...`. To get a list with all arguments use the command `python wjtt.py -h`. For example, the identification model of waterbird can be trained with the command:

```
python wjtt.py --dataset waterbird --id_model True --num_epochs [20,40,60,80,100,120]
```

The default parameters are set to the ones used in the thesis, but they can be specified independently. Our code expects a folder `wjtt/error_sets/waterbird` (or `wjtt/error_sets/celebA`, `wjtt/error_sets/CivilComments`, `wjtt/error_sets/MutliNLI`, depending on which dataset is used) where the error sets are saved. One of the error sets for the above command will then be `nepochs_60_lr_1e-05_batch_size_64_wd_1.0.csv`. To run the second model of JTT for 150 epochs and upweight 100 for this error set, we can use the command

```
wjtt.py --dataset waterbird --num_epochs 150 --weight_func JTT --alpha 100 --error_set_name nepochs_60_lr_1e-05_batch_size_64_wd_1.0.csv
```

To run WJTT with DRO weights and alpha = 8, rho = 2, we can use the command

```
wjtt.py --dataset waterbird --num_epochs 150 --weight_func DRO2 --alpha 8 --rho 2 --error_set_name nepochs_60_lr_1e-05_batch_size_64_wd_1.0.csv
```

## Figures
The python files to generate the figures are as follows:
- Figure 1: `plot_weight_vs_probability.py`
- Figure 2 & 5: `plot_wga_vs_nepochs_id.py`
- Figure 3: `plot_hBarchart_rel_groups.py`
- Figure 4: `plot_unif_groups_comparison.py`
- Figure 6: `plot_ESS_vs_unif.py`

