import dataclasses
import os.path
import pickle
import queue
import threading
import time

import loguru
import numpy as np
from sklearn import ensemble, metrics

from src import configs
from src.utils import common, dao, secrets, kafka_utils


def _initialize_db(ansible: secrets.Ansible, db_config: configs.DbConfig) -> dao.MsSql:
    mssql_creds = ansible.decrypt_yaml(db_config.mssql_creds)
    loguru.logger.info("Loading data from mssql | data table: {}", db_config.data_table)
    return dao.MsSql(**mssql_creds)


def train(config: configs.Config):
    common.set_deterministic_mode(config.seed)

    data_config = config.data
    db_config = config.db
    model_config = config.model
    experiments_config = config.experiments

    if db_config is not None:
        ansible = secrets.Ansible(config.ansible_pwd)
        db = _initialize_db(ansible, db_config)
        loguru.logger.info(
            "Loading data from database | table name: {}", db_config.data_table
        )
        x_train, y_train = db.get_data(db_config.data_table, {"data group": "'train'"})
        x_val, y_val = db.get_data(db_config.data_table, {"data group": "'val'"})

    elif data_config is not None:
        loguru.logger.info("Loading train data from {}", data_config.train_data)
        x_train, y_train, _ = common.load_data_from_csv(
            data_config.train_data, sep=common.CSV_SEPARATOR
        )

        loguru.logger.info("Loading val data from {}", data_config.val_data)
        x_val, y_val, _ = common.load_data_from_csv(
            data_config.val_data, sep=common.CSV_SEPARATOR
        )
    else:
        raise ValueError("Please specify either data_config or db_config")

    train_dir = os.path.join(
        experiments_config.dir, experiments_config.save_to or common.get_current_time()
    )
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    loguru.logger.info("Training...")

    model = ensemble.GradientBoostingClassifier(
        random_state=config.seed, **dataclasses.asdict(model_config)
    )
    model.fit(x_train, y_train)

    pred_val = model.predict(x_val)
    loguru.logger.info("Val accuracy: {}", metrics.accuracy_score(y_val, pred_val))
    loguru.logger.info(
        "Val f1 micro: {}", metrics.f1_score(y_val, pred_val, average="micro")
    )

    save_to = os.path.join(train_dir, "model.sklrn")
    loguru.logger.info("Saving model to {}", save_to)
    with open(save_to, "wb") as model_file:
        pickle.dump(model, model_file)


def eval(config: configs.EvalConfig):
    loguru.logger.info("Loading model from {}", config.model)

    ansible = secrets.Ansible(config.ansible_pwd)

    db_config = config.db
    db = None
    if db_config is not None:
        db = _initialize_db(ansible, db_config)
        x, y = db.get_data(db_config.data_table, {"data group": "'test'"})
    elif config.test_data is not None:
        loguru.logger.info("Loading data from {}", config.test_data)
        x, y, _ = common.load_data_from_csv(config.test_data, sep=common.CSV_SEPARATOR)
    else:
        raise ValueError("Please specify either db config or test-data path")

    kafka_server = ansible.decrypt_yaml(config.kafka_creds)["server"]

    producer = kafka_utils.Producer(kafka_server)
    producer_t = threading.Thread(target=producer.run, args=(x,))
    producer_t.start()

    result = queue.Queue(maxsize=len(x))
    consumer = kafka_utils.Consumer(kafka_server, config.model)
    consumer_t = threading.Thread(target=consumer.run, args=(producer.id, result))
    consumer_t.start()

    producer_t.join()
    consumer_t.join()

    preds = np.empty(len(x))
    for _ in range(len(x)):
        res = result.get()
        preds[res["ind"]] = res["result"]

    f1_micro = metrics.f1_score(y, preds, average="micro")
    accuracy = metrics.accuracy_score(y, preds)

    loguru.logger.info(f"\nF1 micro: {f1_micro}\nAccuracy: {accuracy}")

    if db is not None:
        loguru.logger.info(
            f"Saving metrics into database | table name: {db_config.metrics_table}"
        )
        try:
            db.insert_row(
                db_config.metrics_table, (f"'{config.model}'", accuracy, f1_micro)
            )
        except pyodbc.IntegrityError:
            db.update_row(
                db_config.metrics_table,
                updated_values={"accuracy": accuracy, "f1Micro": f1_micro},
                update_condition={"modelId": f"'{config.model}'"},
            )
