# ConWea: Contextualized Weak Supervision for Text Classification

- [Model](#model)
- [Training](#training)
	- [Required Inputs](#required-inputs)
	- [Commands](#commands)
	- [Requirements](#requirements)
- [Citation](#citation)

## Model

![CONWEA-Framework](docs/ConWea-overview.png)

## Training

### Required inputs
Each Dataset should contain following files:
- **DataFrame pickle file**
  - Example: ```data/nyt/df.pkl```
    - This dataset should contain two columns named ```sentence```, ```label```
    - ```sentence``` contains text and ```label``` contains its corresponding label.
    - Must be named as ```df.pkl```
- **Seed Words Json file**
  - Example: ```data/nyt/seedwords.json```
    - This json file contains seed words list for each label.
    - Must be named as ```seedwords.json```

### Commands


#### Corpus Contextualization: 
The ```contextualize.py``` requires two arguments: ```dataset_path```, which is a path to dataset containing 
required DataFrame, seedwords and ```temp_dir``` is a path to a temporary
directory which is used for dumping intermediate files during contextualizing the corpus.
To contextualize the corpus, please run:
```sh
$ python contextualize.py --dataset_path dataset_path --temp_dir temp_dir_path
```

The ```tests/test_contextualize.py``` is a unittest to check the sanity of contextualization. To run this unittest, please execute:
```shell script
$ python -m unittest tests/test_contextualize.py
``` 
 
#### ConWea - Iterative Framework:
The ```train.py``` requires two arguments: ```dataset_path```, which is a path to dataset containing 
required contextualized corpus DataFrame dumped by ```contextualize.py```, seed words and ```num_iter``` is the
number of iterations for the iterative framework.
To train ConWea, please run:
```shell script
$ python train.py --dataset_path dataset_path --num_iter 5
```

The ```tests/test_conwea.py``` is a unittest to check the sanity of framework. To run this unittest, please execute:
```shell script
$ python -m unittest tests/test_conwea.py
``` 


### Requirements

This project is based on ```python==3.7```. The dependencies are as follow:
```
keras-contrib==2.0.8
scikit-learn==0.21.3
flair==0.4.4
scipy=1.3.1
gensim==3.8.1
numpy==1.17.2
```

## Citation

Citation coming soon!