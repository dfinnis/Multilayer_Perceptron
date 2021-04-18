package multilayer

import (
	// "fmt" //
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func sigmoidLayer(nn neuralNetwork, layer int) {
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = sigmoid(nn.layers[layer].neurons[neuron].value)
		nn.layers[layer].neurons[neuron].outputs = append(nn.layers[layer].neurons[neuron].outputs, nn.layers[layer].neurons[neuron].output)
	}
}

func sigmoid_prime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
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
		// nn.layers[layer].neurons[neuron].outputs = append(nn.layers[layer].neurons[neuron].outputs, ouput[neuron])
	}
}

func softmax_prime(z [][]float64) (d_Z4 [][]float64) {
	for _, sample := range z {
		// fmt.Printf("%v: %v\n", i, sample) ///////////
		soft := softmax(sample)
		var minus []float64
		minus = append(minus, 1 - soft[0])
		minus = append(minus, 1 - soft[1])
		// fmt.Printf("%v: %v\n", soft, minus) ////////////////
		var d []float64
		d = append(d, soft[0] * minus[0])
		d = append(d, soft[1] * minus[1])
		d_Z4 = append(d_Z4, d)
		// fmt.Printf("%v: %v: %v\n", soft, minus, d) ////////////////
	}
	return
	// return softmax(z) * (1 - softmax(z))
}
