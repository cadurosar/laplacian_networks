The code is divided in three folders, one for each of the datasets. Inside you will find a file named dataset.py which is used to train on this dataset.

The conda environment used for the experiments can be created by doing:

```
conda env create laplacian.yml
```

You have to manually download imagenet32 from the website and extract the files at the imagenet32/data folder, so that it contains [train_databatch_1,train_databatch_2...train_databatch_10,val_data].

The corruptions for cifar10 are available at 

Each folder has scripts for training Vanilla(V.sh), Parseval(P.sh), Regularizer(R.sh) and Parseval+Regularizer(PR.sh) networks. For example to run 10 Vanilla networks, ranging from seed 5 to seed 15 one would do:

```
./train_scripts/V.sh 5 15
```

After the networks are trained, you have to execute the test_*.py files to run the tests. This will generate a pandas dataframe on the test_results folder. A script called read_results.py is supplied to either read the results as tables or figures (--matplotlib)

The r_alpha test is available for cifar_10 under r_alpha.py. It generates a matplotlib figure

To test the robustness to corruptions you have to first download the corrupted CIFAR-10C dataset from the site specified in Hendrycks 2019 (https://drive.google.com/drive/folders/1YjMkUNWsIrkVqmCaxf2PAXjKP3aeNgsn) and extract the files in the cifar_10/data folder, which will now contain [spatter.pkl...cifar-10-batches-py...gaussian_noise.pkl]

After extracting the files you can run robustness.py which will generate 4 csv files. Each file contains the results under all types of noises and all severities for a given network.


