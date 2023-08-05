#!/bin/sh

date=$(date '+%d-%m-%Y-%H%M')

cd data
zip csv/csv_${date}.zip csv/*.csv
zip raw/raw_${date}.zip raw/*

cd ..
zip data/alldata_${date}.zip -r data/* -x data/*.zip