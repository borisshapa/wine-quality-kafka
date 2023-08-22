import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    """CatBoost hyperparameters."""

    n_estimators: int = dataclasses.field(default=100)
    learning_rate: float = dataclasses.field(default=0.1)
    max_depth: int = dataclasses.field(default=3)


@dataclasses.dataclass
class DataConfig:
    train_data: str = dataclasses.field(default="data/train.csv")
    val_data: str = dataclasses.field(default="data/val.csv")
    test_data: Optional[str] = dataclasses.field(default=None)


@dataclasses.dataclass
class DbConfig:
    mssql_creds: str = dataclasses.field(default="mssql-creds.yml")
    data_table: str = dataclasses.field(default="Wines")
    metrics_table: str = dataclasses.field(default="Metrics")


@dataclasses.dataclass
class ExperimentsConfig:
    dir: str = dataclasses.field(default="experiments")
    save_to: Optional[str] = dataclasses.field(default="latest")


@dataclasses.dataclass
class Config:
    seed: int = dataclasses.field(default=21)
    data: Optional[DataConfig] = dataclasses.field(default=None)
    db: Optional[DbConfig] = dataclasses.field(default=None)
    ansible_pwd: str = dataclasses.field(default="ansible-pwd.txt")
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    experiments: ExperimentsConfig = dataclasses.field(
        default_factory=ExperimentsConfig
    )


@dataclasses.dataclass
class EvalConfig:
    ansible_pwd: str = dataclasses.field(default="ansible-pwd.txt")
    db: Optional[DbConfig] = dataclasses.field(default=None)
    kafka_creds: str = dataclasses.field(default="kafka-creds.yml")
    test_data: Optional[str] = dataclasses.field(default=None)
    model: str = dataclasses.field(default=None)


@dataclasses.dataclass
class InitDbConfig:
    ansible_pwd: str = dataclasses.field(default="ansible-pwd.txt")
    db: DbConfig = dataclasses.field(default_factory=DbConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    init_file: str = dataclasses.field(default="create_db.sql")


@dataclasses.dataclass
class ClearDbConfig:
    ansible_pwd: str = dataclasses.field(default="ansible-pwd.txt")
    db: DbConfig = dataclasses.field(default_factory=DbConfig)
