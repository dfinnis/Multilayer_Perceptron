#### -- Multilayer Perceptron Test -- ####
# The following commands are the same as running:
# go run main.go

#### -- Print Header -- ####
RESET="\x1b[0m"
BOLD="\x1b[1m"
ITALIC="\x1b[3m"

printf "\E[H\E[2J" ## Clear screen
printf $BOLD
printf $ITALIC
echo "Launching Multilayer Perceptron Test...$RESET\n"

## Create Datasets
printf $ITALIC
echo "Running evaluation.py to create data_training.csv & data_test.csv...$RESET\n"
python3 evaluation.py

## Build
go build

## Train
printf $ITALIC
echo "Training on data_training.csv...$RESET\n"
./Multilayer_Perceptron data_training.csv -t -q

## Predict
printf $ITALIC
echo
echo "Predicting on data_test.csv...$RESET\n"
./Multilayer_Perceptron data_test.csv -p -q

## Cleanup
rm Multilayer_Perceptron
rm data_training.csv
rm data_test.csv
