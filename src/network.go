package multilayer

import (
	"math"
	"math/rand"
)

// neuron is a basic perceptron
type neuron struct {
	bias    float32
	weights []float32
	value   float32   // before activation func
	output  float32   // after activation func
	z       []float32 // value for each sample
	outputs []float32 // output for each sample
}

// activationFunc can be sigmoid or softmax
type activationFunc func(nn neuralNetwork, layer int)

// layer contains a number of neurons
type layer struct {
	neurons    []neuron
	activation activationFunc // input & hidden = sigmoid, output = softmax
}

// lossFunc can be Binary cross-entropy / Mean squared error / Root mean squared error
type lossFunc func(outputs, y [][]float32) float32

// neuralNetwork contains all info about the network
type neuralNetwork struct {
	architecture []int
	layers       []layer
	learningRate float32
	epochs       int
	trainLoss    []float32
	testLoss     []float32
	lossFunc     lossFunc
}

// newNeuron initializes a neuron with random weights and zero bias
func newNeuron(nn neuralNetwork, layer int) neuron {
	var weights []float32
	if layer > 0 {
		for i := 0; i < nn.architecture[layer-1]; i++ {
			weights = append(weights, float32(rand.Float64()*math.Sqrt(2/float64(nn.architecture[layer-1]))))
		}
	}
	return neuron{
		weights: weights,
	}
}

// newLayer initializes a layer with neurons & sigmoid activation
func newLayer(nn neuralNetwork, currentLayer int) layer {
	var neurons []neuron
	for i := 0; i < nn.architecture[currentLayer]; i++ {
		neurons = append(neurons, newNeuron(nn, currentLayer))
	}
	return layer{
		neurons:    neurons,
		activation: sigmoidLayer,
	}
}

// getArchitecture joins the input layer to the rest
func getArchitecture(inputLen int, flags flags) []int {
	var architecture []int
	architecture = append(architecture, inputLen)
	architecture = append(architecture, flags.architecture...)
	if !flags.flagQ {
		printArchitecture(architecture)
	}
	return architecture
}

// setLossFunc parses flags -mse & -rmse, default binaryCrossEntropy loss
func setLossFunc(nn neuralNetwork, flags flags) neuralNetwork {
	if flags.flagMSE {
		nn.lossFunc = meanSquaredError
	} else if flags.flagRMSE {
		nn.lossFunc = rootMeanSquaredError
	} else {
		nn.lossFunc = binaryCrossEntropy
	}
	return nn
}

// setEpochs parses flags -e --early stopping & -ep epochs
func setEpochs(nn neuralNetwork, flags flags) neuralNetwork {
	defaultConfig := defaultConfig()
	if flags.flagE && flags.epochs == defaultConfig.epochs { // early stopping, "infinite" training
		nn.epochs = 42000
	} else {
		nn.epochs = flags.epochs
	}
	return nn
}

// buildNN initializes a neural network
func buildNN(inputLen int, flags flags) neuralNetwork {
	nn := neuralNetwork{}
	nn.architecture = getArchitecture(inputLen, flags)
	// Build layer by layer
	var layer int
	for layer = 0; layer < len(nn.architecture); layer++ {
		nn.layers = append(nn.layers, newLayer(nn, layer))
	}

	nn.layers[0].activation = nil                // Input Layer no activation
	nn.layers[layer-1].activation = softmaxLayer // Output Layer Softmax activation

	nn = setLossFunc(nn, flags)
	nn = setEpochs(nn, flags)
	nn.learningRate = flags.learningRate
	return nn
}
