package multilayer

import (
	"fmt"
	"math"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

type activationFunc func(float64) float64

func sigmoid(z float64) float64 {
	return 1/(1+math.Exp(-z))
}

type neuron struct {
	// value float64 // before activation fucå
	bias float64
	weights []float64
	output float64
}

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

func newNeuron(currentLayer int, nn neuralNetwork) neuron {
	var weights []float64
	if currentLayer > 0 {
		for i := 0; i < nn.architecture[currentLayer - 1]; i++ {
			weights = append(weights, rand.Float64())
		}
	}
	return neuron {
		// value:				rand.Float64(),
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
		// length:				nn.architecture[currentLayer],
		neurons:			neurons,
		activation:			sigmoid,
	}
}

func buildNN(architecture []int) neuralNetwork {
	nn := neuralNetwork{}
	nn.architecture = architecture
	nn.learningRate = 0.01

	for layer := 0; layer < len(architecture); layer++ {
		nn.layers = append(nn.layers, newLayer(layer, nn))
	}
	return nn
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	rand.Seed(time.Now().UnixNano())
	// architecture := []int {16, 16, 16, 2}
	architecture := []int {16, 16, 16, 2}
	nn := buildNN(architecture)

	// printNN
	fmt.Println(nn.learningRate)
	fmt.Println(nn.architecture)
	for i := 0; i < len(nn.architecture); i++ {
		fmt.Println()
		fmt.Println(nn.layers[i])
	}
	fmt.Println()
	// fmt.Println(nn.layers[0].activation)
	// fmt.Println(nn.layers[0].activation(2))
}
