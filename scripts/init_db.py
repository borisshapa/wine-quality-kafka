import copy
from typing import Any

import loguru
import pyrallis

from src import configs
from src.utils import dao, secrets, common


def _insert_data_from_csv_to_db(
    sql: dao.MsSql, csv_file: str, group: str, table_name: str
):
    x, y, _ = common.load_data_from_csv(csv_file, sep=";")
    for x_row, y_row in zip(x, y):
        row = list(x_row) + [y_row, f"'{group}'"]
        sql.insert_row(table_name, tuple(row))


def create_database(creds: dict[str, Any]):
    master_creds = copy.deepcopy(creds)
    master_creds["database"] = "master"
    master_sql_connection = dao.MsSql(**master_creds)
    master_sql_connection.create_database(creds["database"])


@pyrallis.wrap()
def main(config: configs.InitDbConfig):
    ansible = secrets.Ansible(config.ansible_pwd)
    mssql_creds = ansible.decrypt_yaml(config.db.mssql_creds)

    create_database(mssql_creds)
    sql = dao.MsSql(**mssql_creds)

    _, _, header = common.load_data_from_csv(config.data.val_data, sep=";")
    data_attributes = {name: "FLOAT" for name in header}
    data_attributes["quality"] = "TINYINT"
    data_attributes["data group"] = "VARCHAR(32)"
    sql.create_table(config.db.data_table, data_attributes)

    metrics_attributes = {
        "modelId": "NVARCHAR(50) PRIMARY KEY",
        "accuracy": "FLOAT",
        "f1Micro": "FLOAT",
    }
    sql.create_table(config.db.metrics_table, metrics_attributes)

    common_kwargs = {"sql": sql, "table_name": config.db.data_table}
    if config.data.train_data is not None:
        _insert_data_from_csv_to_db(
            csv_file=config.data.train_data, group="train", **common_kwargs
        )
    if config.data.val_data is not None:
        _insert_data_from_csv_to_db(
            csv_file=config.data.val_data, group="val", **common_kwargs
        )
    if config.data.test_data is not None:
        _insert_data_from_csv_to_db(
            csv_file=config.data.test_data, group="test", **common_kwargs
        )


if __name__ == "__main__":
    main()
