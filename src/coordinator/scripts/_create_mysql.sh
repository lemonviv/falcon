#!/bin/bash

export DATA_BASE_PATH=$1

# load variables from properties
. ./deploy/property/db.properties
. ./deploy/property/svc.properties
. coordinator.properties

LOG_FILE_PATH=$DATA_BASE_PATH/logs/start_db.log
{
  (kubectl create configmap mysql-initdb-config --from-file=./deploy/property &> $LOG_FILE_PATH)
    echo "-------------------------- finish creating config map for db --------------------------------"

} || {
echo "--------------------------  creating config map error, check the log, --------------------------------"
}

# create new yaml according template
MYSQL_YAML=./deploy/template/mysql.yaml
cp ./deploy/template/mysql.yaml.template $MYSQL_YAML || exit 1

# create path variables
PV_PATH=$(echo $PV_DB_STORAGE_PATH | sed 's_/_\\/_g')

# replace var in config yaml with customer defined variables
sed -i '' -e "s/MYSQL_CLUSTER_PORT/$MYSQL_CLUSTER_PORT/g" $MYSQL_YAML || exit 1
sed -i '' -e "s/MYSQL_TARGET_PORT/$MYSQL_TARGET_PORT/g" $MYSQL_YAML || exit 1
sed -i '' -e "s/MYSQL_NODE_PORT/$MYSQL_NODE_PORT/g" $MYSQL_YAML || exit 1
sed -i '' -e "s/MYSQL_IMAGE/$MYSQL_IMAGE/g" $MYSQL_YAML || exit 1
sed -i '' -e "s/PV_DB_STORAGE_PATH/$PV_PATH/g" $MYSQL_YAML || exit 1

# apply the job
echo "--------------------------  creating mysql service --------------------------------"
kubectl apply -f $MYSQL_YAML &> $LOG_FILE_PATH|| exit 1

# delete config
rm -f $MYSQL_YAML


