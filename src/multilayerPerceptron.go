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
				nn.layers[layer].neurons[neuron].z = append(nn.layers[layer].neurons[neuron].z, nn.layers[layer].neurons[neuron].value)
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

func compute_loss_prime(outputs [][]float64, y [][]float64) (d_losses [][]float64) {
	for output := 0; output < len(outputs); output++ {
		var b []float64
		b = append(b, y[output][0] / outputs[output][0])
		b = append(b, y[output][1] / outputs[output][1])

		var m []float64
		m = append(m, (1 - y[output][0]) / (1 - outputs[output][0]))
		m = append(m, (1 - y[output][1]) / (1 - outputs[output][1]))

		var d_loss []float64
		d_loss = append(d_loss, b[0] - m[0])
		d_loss = append(d_loss, b[1] - m[1])
		d_losses = append(d_losses, d_loss)
	}
	return
}

func backprop(nn neuralNetwork, output [][]float64, y [][]float64) {
	fmt.Println("oh hi backprop!") /////////////////

	d_A4 := compute_loss_prime(output, y)
	// fmt.Printf("d_A4: %v\n", d_A4) ////////////
	fmt.Printf("len(d_A4): %v\n", len(d_A4)) ////////////
	fmt.Printf("len(d_A4[0]): %v\n", len(d_A4[0])) ////////////
	
	// d_A4[0][0] = -1 ///////////////////
	// d_A4[0][1] = 1 ///////////////////
	
	var z4 [][]float64
	for sample, _ := range output {
		var layer_value []float64
		for _, neuron := range nn.layers[3].neurons {
			layer_value = append(layer_value, neuron.z[sample])
		}
		z4 = append(z4, layer_value)
	}
	prime := softmax_prime(z4)

	var d_z4 [][]float64
	for i, sample := range prime {
		var layer_d []float64
		for j, value := range sample {
			layer_d = append(layer_d, d_A4[i][j] * value)
		}
		d_z4 = append(d_z4, layer_d)
	}
	// fmt.Printf("d_z4[0]: %v\n", d_z4[0]) ////////////
	fmt.Printf("len(d_z4): %v\n", len(d_z4)) ////////////
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

		backprop(nn, output, y)

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
	// fmt.Println()
	// fmt.Println(nn.architecture)
	// for i := 0; i < len(nn.architecture); i++ {
	// 	fmt.Println()
	// 	fmt.Println(nn.layers[i])
	// }
	// fmt.Println()
	// fmt.Println("END!!") ////
}
