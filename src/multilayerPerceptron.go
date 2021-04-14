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
	// value float64 // before activation func
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

func feedforwardNeuron(nn neuralNetwork, layer int, neuron int) {
	fmt.Printf("layer: %v, ", layer) /////////////
	fmt.Printf("neuron: %v\n", neuron) /////////////
	var weightedSum float64
	for weight := 0; weight < len(nn.layers[layer].neurons[neuron].weights); weight++ {
		// fmt.Printf("weight: %v\n", weight) /////////////
		weightedSum += nn.layers[layer - 1].neurons[weight].output * nn.layers[layer].neurons[neuron].weights[weight]
		// fmt.Printf("weightedSum: %v\n", weightedSum) /////////////
	}
	nn.layers[layer].neurons[neuron].output = weightedSum + nn.layers[layer].neurons[neuron].bias
}

func feedforward(nn neuralNetwork) {
	nn.layers[0].neurons[0].output = 0.42 // input layer //////////////////////////////////////////////

	for layer := 1; layer < len(nn.architecture); layer++ {
		// fmt.Printf("layer: %v\n", layer) /////////////
		for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
			// fmt.Printf("neuron: %v\n", neuron) /////////////
			feedforwardNeuron(nn, layer, neuron)
		}
		fmt.Println() ///////////////////
	}
	fmt.Println() /////////////////
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	rand.Seed(time.Now().UnixNano())

	// architecture := []int {16, 16, 16, 2}
	architecture := []int {2, 2, 2} // test architecture
	nn := buildNN(architecture)

	feedforward(nn)

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
