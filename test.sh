#### Multilayer Perceptron Test ####

## Create datasets
python3 evaluation.py

## Build
go build

## Train
./Multilayer_Perceptron data_training.csv -t

## Predict
./Multilayer_Perceptron data_test.csv -p

## Cleanup
rm MultilayerPerceptron
