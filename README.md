# 🍷 Wine Quality

A service for determining the quality of wine by its numerical parameters.

## Structure
* `cd` contains Jenkinsfile for setting up cd pipeline;
* `ci` contains Jenkinsfile for setting up ci pipeline;
* `configs` contains the yaml files that define model hyperparameters;
* `data` ‒ the directory with train/val/test datasets;
* `experiments` contains catboost meta info and a model in binary file;
* `notebooks` ‒ the directory with ipynb files with experiments and data analysis;
* `scripts` ‒ executable files (`python -m scripts.<script-name> [ARGS]`);
* `sh_scripts` ‒ bash scripts helpful for building docker image, run ci/cd pipeline;
* `src` ‒ source code of the function used by scripts, also contains unit test of its functions;
* `tests` ‒ functional tests;

## Instalation
Local installation via pip:
```shell
git clone https://github.com/borisshapa/wine-quality
cd wine-quality
pip install -r requirements.txt
```

Or using docker compose: 
```shell
docker compose build
docker compose up
```

## Data
For data we are using `.csv` format with `;` as separator. Each line looks like this:
```text
0.0;10.4;0.61;0.49;2.1;0.2;5.0;16.0;0.9994;3.16;0.63;8.4;3
```

The parameters go in the following order:
0. wine type (categorical feature: 0 ‒ red wine, 1 ‒ white wine)
1. fixed acidity
2. volatile acidity
3. citric acid
4.  residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol
12. **quality** (score between 0 and 10) ‒ target variable.

The data is downloaded from the source [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
and preprocessed by the script 
```shell
python -m scripts.split_train_val_test --data data/winequality-red.csv,data/winequality-white.csv --val-ratio 0.1 --test-ratio 0.1
```

## Model
The algorithm is based on gradient boosting on decision trees. The service uses the [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
The model is defined by the yaml file. Example:
```yaml
seed: 21
ansible_pwd: "ansible-pwd.txt"

db:
  mssql_creds: "mssql-creds.yml"
  data_table: "Wines"
  metrics_table: "Metrics"

model:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 1

experiments:
  dir: "experiments"
  save_to: "latest"
```

### Training
To run training, use `scripts.train`:
```shell
python -m scripts.train --config_path=configs/db.yaml
```

### Inference
To run model on test dataset use the `scripts.eval`:
```shell
python -m scripts.eval --config_path=configs/eval.yml
```

## Secrets
The credentials to access the database must be encrypted using [ansible-vault](https://docs.ansible.com/ansible/latest/vault_guide/index.html).

Put the database data in a yaml file in the following format:
```shell
server: "mssql,1433"
nuid: "sa"
pwd: "password"
database: "wines"
```

Put the ansible password into the txt file. Use 
```shell
ansible-vault encrypt mssql-creds.yml --vault-password-file ansible-pwd.txt
```
to encrypt the database credentials.

The paths to these files must be passed to scripts using the database.
The config field `ansible_pwd` is responsible for the path to the file with ansible password, and the field `db.mssql_creds` is responsible for the path with encrypted credentials. 

## Kafka

Для использования kafka сервисом модели необходимо создать topic.
Для этого в отдельных сессиях терминала необходимо запустить zookeeper и kafka:

```shell
zookeeper-server-start <path to zookeeper properties>
```

```shell
kafka-server-start <path to kafka properties>
```

Затем создать topic:
```shell
 kafka-topics --bootstrap-server=127.0.0.1:9092 \                                                                
              --create --if-not-exists --topic wine-quality \
              --replication-factor 1 --partitions 1

```