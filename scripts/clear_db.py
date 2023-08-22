import pyrallis

from src import configs
from src.utils import secrets, dao


@pyrallis.wrap()
def main(config: configs.ClearDbConfig):
    ansible = secrets.Ansible(config.ansible_pwd)
    creds = ansible.decrypt_yaml(config.db.mssql_creds)
    sql = dao.MsSql(**creds)

    for table_name in [config.db.data_table, config.db.metrics_table]:
        sql.delete(table_name, {})



if __name__ == "__main__":
    main()
