package multilayer

import (
	"fmt"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

func feedforward(nn neuralNetwork, inputs []float64) {
	// input layer
	for i := 1; i < len(inputs); i++ {
		nn.layers[0].neurons[i - 1].output = inputs[i]
	}

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

func backprop(nn neuralNetwork) {
	fmt.Println("oh hi backprop!") /////////////////
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	rand.Seed(time.Now().UnixNano())

	data := preprocess()
	train_set, test_set := split(data)

	// fmt.Printf("train_set[0]: %v\n", train_set[0]) /////////////////////////////////////////
	// fmt.Printf("test_set[0]: %v\n", test_set[0]) /////////////////////////////////////////
	// fmt.Printf("data:\n %v\n\n", data) /////////////////////////////////////////
	// fmt.Printf("data[0]:\n %v\n\n", data[0]) /////////////////////////////////////////
	fmt.Printf("\n\nlen(data): %v\n", len(data)) /////////////////////////////////////////
	fmt.Printf("len(train_set): %v\n", len(train_set)) /////////////////////////////////////////
	fmt.Printf("len(test_set): %v\n", len(test_set)) /////////////////////////////////////////

	// architecture := []int {len(data[0]) - 1, 16, 16, 16, 2}
	architecture := []int {len(data[0]) - 1, 2, 2, 2} // test architecture ////
	nn := buildNN(architecture)

	// feedforward(nn)
	feedforward(nn, train_set[0])

	// backprop(nn)

	// ## printNN
	fmt.Println(nn.learningRate)
	fmt.Println(nn.architecture)
	for i := 0; i < len(nn.architecture); i++ {
		fmt.Println()
		fmt.Println(nn.layers[i])
	}
	fmt.Println()
}
