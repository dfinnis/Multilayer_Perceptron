package multilayer

import (
	"fmt"
	"math"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

type activationFunc func(nn neuralNetwork, layer int)

type neuron struct {
	bias float64
	weights []float64
	value float64 // before activation func
	output float64 // after activation func
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
			weights = append(weights, rand.Float64() * math.Sqrt(2/float64(nn.architecture[currentLayer - 1])))
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
		activation:			sigmoidLayer,
	}
}

func buildNN(architecture []int) neuralNetwork {
	nn := neuralNetwork{}
	nn.architecture = architecture
	nn.learningRate = 0.01

	var layer int
	for layer = 0; layer < len(architecture); layer++ {
		nn.layers = append(nn.layers, newLayer(layer, nn))
	}
	nn.layers[layer - 1].activation = softmaxLayer
	return nn
}


func sigmoid(z float64) float64 {
	return 1/(1+math.Exp(-z))
}

func sigmoidLayer(nn neuralNetwork, layer int) {
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = sigmoid(nn.layers[layer].neurons[neuron].value)
	}
}

func softmax(values []float64) []float64 {
	max := values[0]
	for _, value := range values {
		max = math.Max(max, value)
	}

	outputs := make([]float64, len(values))
	var sum float64
	for i, value := range values {
		outputs[i] -= math.Exp(value - max)
		sum += outputs[i]
	}
	for i, output := range outputs {
		outputs[i] = output / sum
	}
	return outputs
}

func softmaxLayer(nn neuralNetwork, layer int) {
	// fmt.Printf("layer: %v\n", layer) ////////////////////////////
	var values []float64
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		// fmt.Printf("neuron: %v\n", neuron) ///////////////////////////
		values = append(values, nn.layers[layer].neurons[neuron].value)
	}
	// fmt.Printf("values: %v\n", values) ///////////////////////////
	ouput := softmax(values)
	// fmt.Printf("ouput: %v\n", ouput) ///////////////////////////
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = ouput[neuron]
	}
}

func feedforward(nn neuralNetwork) {
	nn.layers[0].neurons[0].output = 0.42 // input layer //////////////////////////////////////////////

	for layer := 1; layer < len(nn.architecture); layer++ {
		// fmt.Printf("layer: %v\n", layer) /////////////
		for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
			// fmt.Printf("neuron: %v\n", neuron) /////////////
			perceptron := nn.layers[layer].neurons[neuron] // rm for speed? just for human reading
			var weightedSum float64
			for weight := 0; weight < len(perceptron.weights); weight++ {
				weightedSum += nn.layers[layer - 1].neurons[weight].output * perceptron.weights[weight]
			}
			nn.layers[layer].neurons[neuron].value = weightedSum + perceptron.bias
		}
		nn.layers[layer].activation(nn, layer)
		// fmt.Println() ///////////////////
	}
	// fmt.Println() /////////////////
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	rand.Seed(time.Now().UnixNano())

	// architecture := []int {16, 16, 16, 16, 2}
	architecture := []int {2, 2, 2, 2} // test architecture ////
	nn := buildNN(architecture)

	feedforward(nn)

	// ## printNN
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
