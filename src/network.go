package multilayer

import (
	"math"
	"math/rand"
)

type neuron struct {
	bias float64
	weights []float64
	value float64 // before activation func
	output float64 // after activation func
}

type activationFunc func(nn neuralNetwork, layer int)

type layer struct {
	// label string
	// length int
	neurons []neuron
	activation activationFunc // input & hidden = sigmoid, output = softmax
}

type neuralNetwork struct {
	architecture []int
	layers []layer
	learningRate float64
}

func newNeuron(nn neuralNetwork, layer int) neuron {
	var weights []float64
	if layer > 0 {
		for i := 0; i < nn.architecture[layer - 1]; i++ {
			weights = append(weights, rand.Float64() * math.Sqrt(2/float64(nn.architecture[layer - 1])))
		}
	}
	return neuron {
		weights:			weights,
	}
}

func newLayer(nn neuralNetwork, currentLayer int) layer {
	var neurons []neuron
	for i := 0; i < nn.architecture[currentLayer]; i++ {
		neurons = append(neurons, newNeuron(nn, currentLayer))
	}
	return layer {
		// length:				nn.architecture[currentLayer], ////////////
		neurons:			neurons,
		activation:			sigmoidLayer,
	}
}

func buildNN(architecture []int) neuralNetwork {
	nn := neuralNetwork{}
	nn.architecture = architecture
	nn.learningRate = 0.01

	var layer int
	for layer = 0; layer < len(architecture); layer++ {
		nn.layers = append(nn.layers, newLayer(nn, layer))
	}
	nn.layers[layer - 1].activation = softmaxLayer
	return nn
}
