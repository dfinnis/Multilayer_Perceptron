#### -- Multilayer Perceptron Test -- ####
# The following commands are the same as running:
# go run main.go

#### -- Print Header -- ####
RESET="\x1b[0m"
BRIGHT="\x1b[1m"

printf "\E[H\E[2J" ## Clear screen
printf $BRIGHT
echo "Launching Multilayer Perceptron Test...$RESET\n"

## Create Datasets
echo "Running evaluation.py to create data_training.csv and data_test.csv...\n"
python3 evaluation.py

## Build
go build

## Train
echo "Training on data_training.csv...\n"
./Multilayer_Perceptron data_training.csv -t

## Predict
echo
echo "Predicting on data_test.csv...\n"
./Multilayer_Perceptron data_test.csv -p

## Cleanup
rm Multilayer_Perceptron
rm data_training.csv
rm data_test.csv
