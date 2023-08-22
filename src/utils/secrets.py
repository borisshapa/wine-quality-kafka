from ansible_vault import Vault


class Ansible:
    def __init__(self, pwd_filename: str):
        with open(pwd_filename, "r") as pwd_file:
            self.vault = Vault(pwd_file.read())

    def decrypt_yaml(self, yaml_filename: str) -> dict[str, str]:
        with open(yaml_filename, "r") as yaml_file:
            data = self.vault.load(yaml_file.read())
        return data
