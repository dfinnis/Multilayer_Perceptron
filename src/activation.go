package multilayer

import (
	// "fmt"
	"math"
)

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
	var values []float64
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		values = append(values, nn.layers[layer].neurons[neuron].value)
	}
	ouput := softmax(values)
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = ouput[neuron]
	}
}
