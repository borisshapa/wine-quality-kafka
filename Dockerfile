FROM python:3.10

SHELL ["/bin/bash", "-c"]

ARG ANSIBLE_PWD
ARG MSSQL_SERVER
ARG MSSQL_UID
ARG MSSQL_PWD
ARG MSSQL_DATABASE
ARG KAFKA_SERVER

RUN pip install --upgrade pip

WORKDIR /app
ADD . /app

RUN chmod +x sh_scripts/wait_for_it.sh sh_scripts/install_odbc_debian.sh
RUN ./sh_scripts/install_odbc_debian.sh

RUN printf "server: \"$MSSQL_SERVER\"\nuid: \"$MSSQL_UID\"\npwd: \"$MSSQL_PWD\" \
    \ndatabase: \"$MSSQL_DATABASE\"" >> mssql-creds.yml

RUN printf "server: \"$KAFKA_SERVER\"" >> kafka-creds.yml

RUN printf "$ANSIBLE_PWD" >> ansible-pwd.txt

RUN python3 -m pip install --user ansible
RUN printf "export PATH=$HOME/.local/bin:$PATH\n" >> ~/.bashrc
RUN source ~/.bashrc &&  \
    ansible-vault encrypt mssql-creds.yml --vault-password-file ansible-pwd.txt && \
    ansible-vault encrypt kafka-creds.yml --vault-password-file ansible-pwd.txt

RUN pip install -r requirements.txt