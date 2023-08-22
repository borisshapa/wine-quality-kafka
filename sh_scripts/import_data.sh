#!/bin/bash

for i in {1..50};
do
    /opt/mssql-tools18/bin/sqlcmd -S ${MSSQL_SERVER} -U ${MSSQL_UID} -P ${MSSQL_PWD} -i /app/sh_scripts/create_db.sql
    if [ $? -eq 0 ]
    then
        echo "setup.sql completed"
        break
    else
        echo "not ready yet..."
        sleep 10
    fi
done

# add new group colum with data group
data_groups=("train" "val" "test")
for data_group in ${data_groups[@]}
do
  awk "BEGIN{ FS = OFS = \";\" } { print \$0, (NR==1? \"data group\" : \"${data_group}\") }" data/${data_group}.csv > data/${data_group}_with_group.csv
  /opt/mssql-tools18/bin/bcp WineQuality.dbo.Wines in "data/${data_group}_with_group.csv" -c -t';' -F2 -S ${MSSQL_SERVER} -U ${MSSQL_UID} -P ${MSSQL_PWD}
done
