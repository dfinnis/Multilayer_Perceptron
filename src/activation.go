package multilayer

import "math"

func sigmoid(z float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-z))))
}

func sigmoidLayer(nn neuralNetwork, layer int) {
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = sigmoid(nn.layers[layer].neurons[neuron].value)
		nn.layers[layer].neurons[neuron].outputs = append(nn.layers[layer].neurons[neuron].outputs, nn.layers[layer].neurons[neuron].output)
	}
}

func sigmoid_prime(inputs [][]float32) [][]float32 {
	outputs := make([][]float32, len(inputs))
	for i, input := range inputs {
		outputs[i] = make([]float32, len(inputs[0]))
		for j, z := range input {
			outputs[i][j] = sigmoid(z) * (1 - sigmoid(z))
		}
	}
	return outputs
}

func max32(max, value float32) float32 {
	if max > value {
		return max
	}
	return value
}

func softmax(values []float32) []float32 {
	max := values[0]
	for _, value := range values {
		max = max32(max, value)
	}

	outputs := make([]float32, len(values))
	var sum float32
	for i, value := range values {
		outputs[i] -= float32(math.Exp(float64(value - max)))
		sum += outputs[i]
	}
	for i, output := range outputs {
		outputs[i] = output / sum
	}
	return outputs
}

func softmaxLayer(nn neuralNetwork, layer int) {
	var values []float32
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		values = append(values, nn.layers[layer].neurons[neuron].value)
	}
	ouput := softmax(values)
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = ouput[neuron]
	}
}

func softmax_prime(z [][]float32) (d_Z [][]float32) {
	for _, sample := range z {
		soft := softmax(sample)
		var minus []float32
		minus = append(minus, 1-soft[0])
		minus = append(minus, 1-soft[1])
		var d []float32
		d = append(d, soft[0]*minus[0])
		d = append(d, soft[1]*minus[1])
		d_Z = append(d_Z, d)
	}
	return
}
