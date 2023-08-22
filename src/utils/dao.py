from typing import Any

import loguru
import numpy as np
import pyodbc

from src.utils import common


class MsSql:
    def __init__(self, server: str, uid: str, pwd: str, database: str):
        connection_string = (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={server};"
            f"Database={database};"
            f"uid={uid};"
            f"pwd={pwd};"
            "TrustServerCertificate=yes;"
        )
        connection = pyodbc.connect(connection_string, autocommit=True)
        self.cursor = connection.cursor()

    def create_database(self, db_name: str):
        loguru.logger.info("Creating database {}", db_name)
        self.cursor.execute(
            f"""IF NOT EXISTS(SELECT * FROM sys.databases WHERE name='{db_name}') BEGIN
                    CREATE DATABASE {db_name};
                END"""
        )

    def create_table(self, table_name: str, attributes: dict[str, str]):
        loguru.logger.info(
            "Creating table {} with attributes {}", table_name, attributes
        )
        attributes_cmd = ", ".join(
            f"[{name}] {spec}" for name, spec in attributes.items()
        )
        self.cursor.execute(
            f"""IF NOT EXISTS(SELECT * FROM sysobjects WHERE name='{table_name}' and xtype='U') BEGIN
                    CREATE TABLE {table_name}({attributes_cmd});
                END"""
        )

    def get_data(
        self, table_name: str, condition: dict[str, Any] | None = None
    ) -> tuple[common.Features, common.Target]:
        request_builder = [f"SELECT * FROM {table_name}"]
        if condition is not None:
            where_str = common.get_condition_str(condition)
            request_builder.append(f" WHERE {where_str}")
        request_builder.append(";")
        request = "".join(request_builder)

        self.cursor.execute(request)
        data = self.cursor.fetchall()

        x, y = [], []
        for row in data:
            x.append(row[:-2])
            y.append(row[-2])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def insert_row(self, table_name: str, row: tuple[Any, ...]):
        row_str = ", ".join(map(str, row))
        self.cursor.execute(f"""INSERT INTO {table_name} VALUES ({row_str});""")

    def update_row(
        self,
        table_name: str,
        update_condition: dict[str, Any],
        updated_values: dict[str, Any],
    ):
        set_str = ", ".join(
            [f"[{key}]={value}" for key, value in updated_values.items()]
        )
        where_str = common.get_condition_str(update_condition)
        self.cursor.execute(f"""UPDATE {table_name} SET {set_str} WHERE {where_str}""")

    def delete(self, table_name: str, condition: dict[str, Any]):
        condition_str = common.get_condition_str(condition).strip()
        if condition_str:
            condition_str = f" WHERE {condition_str}"
        self.cursor.execute(f"""IF EXISTS(SELECT * FROM sysobjects WHERE name='{table_name}' and xtype='U') BEGIN
                                    DELETE FROM {table_name}{condition_str}
                                END""")
