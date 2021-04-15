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

func split(data [][]float64) (train_set [][]float64, test_set [][]float64) {
	// Shuffle
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
	// Split
	split := 0.8
	var sample int
	for ; sample < int((float64(len(data)) * split)); sample++ {
		train_set = append(train_set, data[sample])
	}
	for ; sample < len(data); sample++ {
		test_set = append(test_set, data[sample])
	}
	return
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
