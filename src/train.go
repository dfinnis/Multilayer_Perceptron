package multilayer

import (
	"fmt"
)

// setInput loads one samples data to the input layer
func setInput(nn neuralNetwork, inputs [][]float32, sample int) {
	for i := 0; i < len(inputs[0]); i++ {
		nn.layers[0].neurons[i].output = inputs[sample][i]
		nn.layers[0].neurons[i].outputs = append(nn.layers[0].neurons[i].outputs, inputs[sample][i])
	}
}

// feedforwardSample runs one sample through the network
func feedforwardSample(nn neuralNetwork) {
	for layer := 1; layer < len(nn.architecture); layer++ {
		for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
			perceptron := nn.layers[layer].neurons[neuron] // rm for speed? just for human reading
			var weightedSum float32
			for weight := 0; weight < len(perceptron.weights); weight++ {
				weightedSum += nn.layers[layer-1].neurons[weight].output * perceptron.weights[weight]
			}
			nn.layers[layer].neurons[neuron].value = weightedSum + perceptron.bias
			nn.layers[layer].neurons[neuron].z = append(nn.layers[layer].neurons[neuron].z, nn.layers[layer].neurons[neuron].value)
		}
		nn.layers[layer].activation(nn, layer)
	}
}

// getOutput reads the output layer
func getOutput(nn neuralNetwork) []float32 {
	var output []float32
	for neuron := 0; neuron < nn.architecture[len(nn.architecture)-1]; neuron++ {
		output = append(output, nn.layers[len(nn.architecture)-1].neurons[neuron].output)
	}
	return output
}

// feedforward inputs samples & reads the output
func feedforward(nn neuralNetwork, inputs [][]float32) (outputs [][]float32) {
	for sample := 0; sample < len(inputs); sample++ {
		setInput(nn, inputs, sample)
		feedforwardSample(nn)
		outputs = append(outputs, getOutput(nn))
	}
	return
}

// train trains the network & saves the model
func train(nn neuralNetwork, train_set [][]float32, test_set [][]float32, flagE bool) {
	fmt.Printf("\n%v%vTrain model%v\n\n", BRIGHT, UNDERLINE, RESET)
	for epoch := 1; epoch <= nn.epochs; epoch++ {
		input, y := splitXY(train_set)

		output := feedforward(nn, input)
		backprop(nn, output, y)

		trainLoss := computeLoss(output, y)
		nn.trainLoss = append(nn.trainLoss, trainLoss)

		testLoss := predictLoss(nn, test_set)
		nn.testLoss = append(nn.testLoss, testLoss)

		// print validation metrics
		fmt.Printf("\repoch %5v/%v - train loss: %-10v - test loss: %-10v", epoch, nn.epochs, trainLoss, testLoss)

		if epoch > 1000 && flagE {
			if nn.testLoss[len(nn.testLoss)-1] > nn.testLoss[len(nn.testLoss)-2] {
				break
			}
		}
	}
	fmt.Printf("\n\n")
	visualize(nn.trainLoss, nn.testLoss)
	saveModel(nn)
}
