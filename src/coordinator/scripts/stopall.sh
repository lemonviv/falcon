#!/bin/bash

. ./deploy/property/svc.properties


kubectl delete all --all
kubectl delete deploymenbat,svc mysql
kubectl delete pvc mysql-pv-claim
kubectl delete pv mysql-pv-volume
kubectl delete pvc $COORD_STORAGE-pvc
kubectl delete pv $COORD_STORAGE-pv
kubectl delete pvc $LISTENER_STORAGE-pvc
kubectl delete pv $LISTENER_STORAGE-pv

kubectl delete configmap mysql-initdb-config
kubectl delete configmap coord-config
kubectl delete configmap redis-config
kubectl delete configmap redis-envs
kubectl delete configmap listener-config
kubectl delete configmap listener-config-1
kubectl delete configmap listener-config-2

. config_coord.properties
rm -rf $DATA_BASE_PATH/database
rm -rf $DATA_BASE_PATH/run_time_logs/*
rm -rf $DATA_BASE_PATH/logs/*

. config_listener.properties

rm -rf /Users/nailixing/GOProj/src/github.com/falcon/src/coordinator/.falcon_listener_1/*
rm -rf /Users/nailixing/GOProj/src/github.com/falcon/src/coordinator/.falcon_listener_2/*

bash scripts/status.sh user
