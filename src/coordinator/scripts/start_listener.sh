#!/bin/bash

. userdefined.properties

# load env variables
if [ $DATA_BASE_PATH ];then
	echo "LISTENER_BASE_PATH is exist, and echo to = $LISTENER_BASE_PATH"
else
	export DATA_BASE_PATH=$PWD
fi

# if using this scripts, assume running in production
env=prod

title()
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

help()
{
    cat <<- EOF
    Desc: used to launch the services
    Usage: start all services using: bash scripts/kubernetes/start.sh
           start db using: bash scripts/kubernetes/start.sh db
           start web using: bash scripts/kubernetes/start.sh web
EOF
}


start_listener()
{
  title "Starting falcon listener..."
  bash ./scripts/_create_listener.sh $DATA_BASE_PATH $env|| exit 1
  title "falcon coord listener"
}

create_folders()
{
      title "Creating folders"
      mkdir $DATA_BASE_PATH
}

title "Guidence"
help
create_folders

title "Creating the cluster role binding"
# ensure python api in pod has auth to control kubernetes
kubectl create clusterrolebinding add-on-cluster-admin --clusterrole=cluster-admin --serviceaccount=default:default

# Pull images from Docker Hub
# bash $HOST_WORKDIR_PATH/scripts/pull_images.sh || exit 1

start_listener || exit 1
title "Creating Listener Done"
