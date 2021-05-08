# Multilayer Perceptron

A deep neural network that predicts whether a tumor is malignant or benign.

Based on the Wisconsin breast cancer diagnosis dataset.

#### Implemented from scratch:
* Data preprocessing
* Flexible neural network architecture
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

Default behaviour is to split the data into training & test sets, train a model on the training set, show metrics for training & test sets, then save model as *model.json*.


## Flags

### -t --train

```go run main.go -t -s 4242```

While training prints metrics & confusion matrix for training & test sets.
After training shows a line graph of loss over training period.

<img src="https://github.com/dfinnis/Multilayer_Perceptron/blob/master/img/t.gif" width="420">

<img src="https://github.com/dfinnis/Multilayer_Perceptron/blob/master/img/loss.png" width="600">

without *--predict* or *--early* or *--seed*, *--train* uses the entire dataset to train.


### -p --predict

```go run main.go -p -s 4242```

Only predict, don't train. Uses the entire dataset to predict unless seeded with *-s*.

The default model path is "model.json", but with additional argument *FILEPATH* model is loaded from given filepath.

Predict prints metrics & confusion matrix for predictions on the test set.

<img src="https://github.com/dfinnis/Multilayer_Perceptron/blob/master/img/p.png" width="800">


### -s --seed

```go run main.go -e -s 4242```

Provide addtional argument SEED integer for randomization.

This seeds the pseudo-randomization of weights and shuffling of data.
Thus a data split, model & loss can be replicated exactly with a given seed.
The default seed is the current time.


### -e --early

```go run main.go -e```

Early stopping. Stop training when test set loss starts to increase. This avoids overfitting & minimizes test loss.


### -ep --epochs

```go run main.go -ep 14000```

Provide addtional argument EPOCHS to determine length of training. Must be an integer between 0 & 100000.

The default number of epochs is 15000, which is usually around when test loss reaches minimum on default settings.

### -l --learning

```go run main.go -e -l 0.1```

Provide addtional argument LEARNING rate. Must be a float between 0 & 1. The default learning rate is 0.01.


### -a --architecture

```go run main.go -e -a "30 30 2"```

Provide addtional argument ARCHITECTURE as string.

The default architecture is "16 16 2". 2 hidden layers with 16 neurons, and an output layer with 2 neurons.

<img src="https://github.com/dfinnis/Multilayer_Perceptron/blob/master/img/a.png" width="320">


### -mse --mean

```go run main.go -mse```

While training print loss metric: Mean squared error. Default loss metric: Binary cross entropy log loss.


### -rmse --root

```go run main.go -rmse```

While training print loss metric: Root mean squared error.


### -q --quiet

```go run main.go -q```

Don't print architecture or seed or additional metrics while training.


### data.csv

```go run main.go data.csv```

Any non-flag argument will be read as data path. The default data path is *data.csv*.


## Test

42 provides *evaluation.py* which splits *data.csv* randomly into *data_training.csv* & *data_test.csv*.

*test.sh* runs *evaluation.py* then trains a model with *data_training.csv*, & predicts with *data_test.csv*.

```./test.sh```

<img src="https://github.com/dfinnis/Multilayer_Perceptron/blob/master/img/test.png" width="840">


## model.json

The model is saved every training epoch in json format as a 2d array of neurons with bias & weights. Below is the first 42 lines of a model starting to train. We see the first neuron in the first hidden layer, it has 30 weights corresponding to the 30 neurons in the input layer.

<img src="https://github.com/dfinnis/Multilayer_Perceptron/blob/master/img/model.gif">


## reset_cursor.sh

The cursor is hidden while training for clean output.
If Multilayer_Perceptron is killed, you may need to reset your cursor.

```./reset_cursor.sh```

