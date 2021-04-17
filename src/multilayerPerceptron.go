package multilayer

import (
	"fmt"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

func feedforward(nn neuralNetwork, inputs [][]float64) (outputs [][]float64) {
	for sample := 0; sample < len(inputs); sample++ {
		// Input
		for i := 0; i < len(inputs[0]); i++ {
			nn.layers[0].neurons[i].output = inputs[sample][i]
		}
		// Feedforward
		for layer := 1; layer < len(nn.architecture); layer++ {
			for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
				perceptron := nn.layers[layer].neurons[neuron] // rm for speed? just for human reading
				var weightedSum float64
				for weight := 0; weight < len(perceptron.weights); weight++ {
					weightedSum += nn.layers[layer - 1].neurons[weight].output * perceptron.weights[weight]
				}
				nn.layers[layer].neurons[neuron].value = weightedSum + perceptron.bias
			}
			nn.layers[layer].activation(nn, layer)
		}
		// Output
		var output []float64
		for neuron := 0; neuron < nn.architecture[len(nn.architecture) - 1]; neuron++ {
			output = append(output, nn.layers[len(nn.architecture) - 1].neurons[neuron].output)
		}
		outputs = append(outputs, output)
	}
	return
}

func backprop(nn neuralNetwork) {
	fmt.Println("oh hi backprop!") /////////////////
}

func train(nn neuralNetwork, train_set [][]float64) {

	for epoch := 1; epoch <= nn.epochs; epoch++ {
		// shuffle(train_set)
		input, y := split_x_y(train_set)
		// fmt.Printf("input: %v\n", input) ////////////
		fmt.Printf("len(input): %v\n", len(input)) ////////////
		fmt.Printf("len(input[0]): %v\n", len(input[0])) ////////////
		// fmt.Printf("y: %v\n", y) ////////////
		fmt.Printf("len(y): %v\n", len(y)) ////////////
		fmt.Printf("len(y[0]): %v\n", len(y[0])) ////////////

		output := feedforward(nn, input)
		// fmt.Printf("output: %v\n", output) /////////////
		fmt.Printf("len(output): %v\n", len(output)) /////////////
		fmt.Printf("len(output[0]): %v\n", len(output[0])) /////////////

		// backprop(nn)

		// train_loss = compute_loss(output, y)
		// train_losses.append(train_losses, train_loss)
		// predict(test_set)
		// test_loss = compute_loss(output, y)
		// test_losses.append(test_losses, test_loss)

		// print validation metrics
		fmt.Printf("epoch %v/%v: train loss = %v, test loss = %v\n", epoch, nn.epochs, "??", "??")
	}
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {

	rand.Seed(time.Now().UnixNano())

	data := preprocess()
	train_set, test_set := split(data)

	fmt.Printf("\n\nlen(data): %v\n", len(data)) /////////////////////////////////////////
	fmt.Printf("len(train_set): %v\n", len(train_set)) /////////////////////////////////////////
	fmt.Printf("len(test_set): %v\n", len(test_set)) /////////////////////////////////////////

	// architecture := []int {len(data[0]) - 1, 16, 16, 2}
	architecture := []int {len(data[0]) - 1, 2, 2, 2} // test architecture ////
	nn := buildNN(architecture)

	train(nn, train_set)

	// ## printNN
	// fmt.Println(nn.learningRate)
	fmt.Println()
	fmt.Println(nn.architecture)
	for i := 0; i < len(nn.architecture); i++ {
		fmt.Println()
		fmt.Println(nn.layers[i])
	}
	fmt.Println()
}
