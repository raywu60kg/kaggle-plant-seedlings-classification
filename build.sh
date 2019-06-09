#!/bin/sh
BASE=$(dirname $0)
cd $BASE/

TAG="kaggle-plant-seedling-classification:dev"
docker build -t=$TAG ./


