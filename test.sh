#### -- Multilayer Perceptron Test -- ####
# The following commands are the same as running:
# go run main.go

## Create Datasets
python3 evaluation.py

## Build
go build

## Train
./Multilayer_Perceptron data_training.csv -t -e

## Predict
./Multilayer_Perceptron data_test.csv -p

## Cleanup
rm Multilayer_Perceptron
rm data_training.csv
rm data_test.csv
