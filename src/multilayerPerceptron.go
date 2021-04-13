package multilayer

import (
	"fmt"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

type neuron struct {
	value float64
	bias float64
	weights []float64
}

type layer struct {
	label string
	length int
	neurons []neuron
	// activation function_pointer (sigmoid / softmax)
}

type neuralNetwork struct {
	architecture []int
	layers []layer
	learningRate float64
}

func newNeuron(currentLayer int, nn neuralNetwork) neuron {
	var weights []float64
	if currentLayer > 0 {
		for i := 0; i < nn.architecture[currentLayer - 1]; i++ {
			weights = append(weights, rand.Float64())
		}
	}
	return neuron {
		value:				rand.Float64(),
		weights:			weights,
	}
}

func newLayer(currentLayer int, nn neuralNetwork) layer {
	// var label string
	// if layer == 0 {
	// 	label = "input"
	// }
	var neurons []neuron
	for i := 0; i < nn.architecture[currentLayer]; i++ {
		neurons = append(neurons, newNeuron(currentLayer, nn))
	}
	return layer {
		length:				nn.architecture[currentLayer],
		neurons:			neurons,
	}
}

func buildNN(architecture []int) neuralNetwork {
	nn := neuralNetwork{}
	nn.learningRate = 0.01
	nn.architecture = architecture

	for layer := 0; layer < len(architecture); layer++ {
		nn.layers = append(nn.layers, newLayer(layer, nn))
	}
	return nn
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	rand.Seed(time.Now().UnixNano())
	architecture := []int {16, 16, 16, 2}
	nn := buildNN(architecture)

	// printNN
	fmt.Println(nn.learningRate)
	fmt.Println(nn.architecture)
	for i := 0; i < len(nn.architecture); i++ {
		fmt.Println()
		fmt.Println(nn.layers[i])
	}
}
