import unittest

import numpy as np

from src.utils import dao, secrets


class TestDao(unittest.TestCase):
    def setUp(self) -> None:
        ansible = secrets.Ansible("ansible-pwd.txt")
        mssql_creds = ansible.decrypt_yaml("mssql-creds.yml")
        self.sql = dao.MsSql(**mssql_creds)

    def test_insert(self):
        table_name = "Wines"
        row = (
            1.0,
            6.6,
            0.25,
            0.25,
            1.3,
            0.04,
            28.0,
            85.0,
            0.98984,
            2.87,
            0.48,
            11.2,
            6,
            "new data group",
        )

        self.sql.insert_row(table_name, row)
        x, y = self.sql.get_data(
            table_name, condition={"data group": "'new data group'"}
        )

        self.sql.delete(table_name, {"data group": "'new data group'"})
        empty_x, empty_y = self.sql.get_data(
            table_name, condition={"data group": "'new data group'"}
        )

        np.testing.assert_array_equal(np.array(row[:-2], dtype=np.float32), x)
        np.testing.assert_array_equal(np.array([row[-2]], dtype=np.float32), y)
        self.assertEqual(len(empty_x), 0)
        self.assertEqual(len(empty_y), 0)

    def test_update(self):
        table_name = "Metrics"
        model_id = "'model1'"
        row = (model_id, 0.15, 0.943)

        self.sql.insert_row(table_name, row)
        self.sql.update_row(
            table_name=table_name,
            update_condition={"modelId": model_id},
            updated_values={"accuracy": 0.8},
        )
        df = self.sql.get_data("Metrics", condition={"modelId": model_id})
        rows = list(df.itertuples(index=False, name=None))
        first_row = rows[0]

        self.sql.delete(table_name, {"modelId": model_id})
        empty_df = self.sql.get_data("Metrics", condition={"modelId": model_id})

        self.assertEqual(len(rows), 1)
        self.assertEqual(first_row[1], 0.8)
        self.assertEqual(first_row[2], 0.943)
        self.assertTrue(empty_df.empty)
