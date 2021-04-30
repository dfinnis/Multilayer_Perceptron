# Multilayer_Perceptron

A deep neural network that predicts whether a tumor is malignant or benign.

Based on the Wisconsin breast cancer diagnosis dataset.

#### Implemented from scratch:
* Data preprocessing
* Feedforward
* Backpropagation
* Gradient descent
* Sigmoid & Softmax activation functions
* Binary cross-entropy loss
* Matrix multiplication

See the [subject](https://github.com/dfinnis/Multilayer_Perceptron/blob/master/subject.pdf) for more details.

## Getting Started

First you need to have your golang workspace set up on your machine.
Then clone this repo into your go-workspace/src/ folder. <br>
```git clone https://github.com/dfinnis/Multilayer_Perceptron.git; cd Multilayer_Perceptron```

Download dependencies. <br>
```go get -d ./...; pip install -r requirements.txt```

To run. <br>
```go run main.go```

Alternatively, build and run the binary. <br>
```go build; ./Multilayer_Perceptron```

Default behaviour is if model.json exists only predict, if model.json does not exist train & predict.

## Flags

### -t --train

Only train, don't predict. Uses the entire dataset to train. Overwrites existing model.json.

### -p --predict

Only predict, don't train. Uses the entire dataset to predict unless seeded with *-s*.

The default model path is "model.json", but with additional argument *FILEPATH* model is loaded from given filepath.

*-t -p* both train & predict, even if model.json already exists.

### -e --early

```go run main.go -t -p -e```

Early stopping. Stop training when test set loss starts to increase. This avoids overfitting.

### -ep --epochs

```go run main.go -t -p -ep 14000```

Provide addtional argument EPOCHS to determine length of training. Must be an integer between 0 & 100000.

The default number of epochs is 15000, which is usually around when test loss reaches its minimum on default settings.

### -l --learning

```go run main.go -t -p -e -l 0.1```

Provide addtional argument LEARNING rate. Must be a float between 0 & 1. The default learning rate is 0.01.

### -a --architecture

```go run main.go -t -p -e -a "30 30 2"```

Provide addtional argument ARCHITECTURE as string.

The default architecture is "16 16 2".

### -s --seed

```go run main.go -t -p -e -s 4242```

Provide addtional argument SEED integer for randomization.

This seeds the pseudo-randomization of weights and shuffling of data.
Thus a data split, model & loss can be replicated exactly with a given seed.

## Test

```./test.sh```

42 provides *evaluation.py* which splits *data.csv* randomly into *data_training.csv* & *data_test.csv*.

*test.sh* runs *evaluation.py* then trains a model with *data_training.csv*, & predicts with *data_test.csv*.
