package multilayer

import (
	"fmt"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

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

func backprop(nn neuralNetwork) {
	fmt.Println("oh hi backprop!") /////////////////
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	data := preprocess()
	// fmt.Printf("data:\n %v\n\n", data) /////////////////////////////////////////

	fmt.Printf("\n\nlen(data): %v\n\n", len(data)) /////////////////////////////////////////

	fmt.Printf("len(data[0]): %v\n\n", len(data[0])) /////////////////////////////////////////



	rand.Seed(time.Now().UnixNano())

	// architecture := []int {16, 16, 16, 16, 2}
	architecture := []int {2, 2, 2, 2} // test architecture ////
	nn := buildNN(architecture)

	feedforward(nn)

	// backprop(nn)

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
