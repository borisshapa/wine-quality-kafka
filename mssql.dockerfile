FROM mcr.microsoft.com/mssql/server

ARG MSSQL_UID
ARG MSSQL_PWD

ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=${MSSQL_PWD}
ENV USER_ID=${MSSQL_UID}

USER root

CMD ./opt/mssql/bin/sqlservr