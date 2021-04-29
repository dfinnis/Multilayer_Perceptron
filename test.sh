#### -- Multilayer Perceptron Test -- ####

## Create Datasets
python3 evaluation.py

## Build
go build

## Train
./Multilayer_Perceptron data_training.csv -t

## Predict
./Multilayer_Perceptron data_test.csv -p

## Cleanup
rm MultilayerPerceptron
rm data_training.csv
rm data_test.csv