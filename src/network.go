package multilayer

import (
	"math"
	"math/rand"
)

type neuron struct {
	bias    float64
	weights []float64
	value   float64   // before activation func
	output  float64   // after activation func
	z       []float64 // value for each sample
	outputs []float64 // output for each sample
}

type activationFunc func(nn neuralNetwork, layer int)

type layer struct {
	neurons    []neuron
	activation activationFunc // input & hidden = sigmoid, output = softmax
}

type neuralNetwork struct {
	architecture []int
	layers       []layer
	learningRate float64
	epochs       int
	trainLoss    []float64
	testLoss     []float64
}

// newNeuron initializes a neuron with random weights and zero bias
func newNeuron(nn neuralNetwork, layer int) neuron {
	var weights []float64
	if layer > 0 {
		for i := 0; i < nn.architecture[layer-1]; i++ {
			weights = append(weights, rand.Float64()*math.Sqrt(2/float64(nn.architecture[layer-1])))
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
func getArchitecture(inputLen int, arch []int) []int {
	var architecture []int
	architecture = append(architecture, inputLen)
	architecture = append(architecture, arch...)
	printArchitecture(architecture)
	return architecture
}

// buildNN initializes a neural network
func buildNN(inputLen int, architecture []int, flagE bool) neuralNetwork {
	nn := neuralNetwork{}
	nn.architecture = getArchitecture(inputLen, architecture)
	nn.learningRate = 0.01

	var layer int
	for layer = 0; layer < len(nn.architecture); layer++ {
		nn.layers = append(nn.layers, newLayer(nn, layer))
	}
	nn.layers[0].activation = nil
	nn.layers[layer-1].activation = softmaxLayer
	if flagE { // early stopping, "infinite" training
		nn.epochs = 42000
	} else {
		nn.epochs = 15000
	}
	return nn
}
