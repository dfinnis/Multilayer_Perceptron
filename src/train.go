package multilayer

import "fmt"

func feedforward(nn neuralNetwork, inputs [][]float64) (outputs [][]float64) {
	for sample := 0; sample < len(inputs); sample++ {
		// Input
		for i := 0; i < len(inputs[0]); i++ {
			nn.layers[0].neurons[i].output = inputs[sample][i]
			nn.layers[0].neurons[i].outputs = append(nn.layers[0].neurons[i].outputs, inputs[sample][i])
		}
		// Feedforward
		for layer := 1; layer < len(nn.architecture); layer++ {
			for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
				perceptron := nn.layers[layer].neurons[neuron] // rm for speed? just for human reading
				var weightedSum float64
				for weight := 0; weight < len(perceptron.weights); weight++ {
					weightedSum += nn.layers[layer-1].neurons[weight].output * perceptron.weights[weight]
				}
				nn.layers[layer].neurons[neuron].value = weightedSum + perceptron.bias
				nn.layers[layer].neurons[neuron].z = append(nn.layers[layer].neurons[neuron].z, nn.layers[layer].neurons[neuron].value)
			}
			nn.layers[layer].activation(nn, layer)
		}
		// Output
		var output []float64
		for neuron := 0; neuron < nn.architecture[len(nn.architecture)-1]; neuron++ {
			output = append(output, nn.layers[len(nn.architecture)-1].neurons[neuron].output)
		}
		outputs = append(outputs, output)
	}
	return
}

func train(nn neuralNetwork, train_set [][]float64, test_set [][]float64) {
	fmt.Printf("%vTraining model...%v\n", BRIGHT, RESET)
	for epoch := 1; epoch <= nn.epochs; epoch++ {
		shuffle(train_set)
		input, y := split_x_y(train_set)

		output := feedforward(nn, input)
		backprop(nn, output, y)

		trainLoss := computeLoss(output, y)
		nn.trainLoss = append(nn.trainLoss, trainLoss)

		testLoss := predictLoss(nn, test_set)
		nn.testLoss = append(nn.testLoss, testLoss)

		// print validation metrics
		fmt.Printf(" epoch %5v/%v - train loss: %-18v - test loss: %v\r", epoch, nn.epochs, trainLoss, testLoss)
	}
	fmt.Printf("\n\n")
	visualize(nn.trainLoss, nn.testLoss)
	saveModel(nn)
}
