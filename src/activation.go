package multilayer

import "math"

// max32 returns the maximum value from 2 inputs
func max32(max, value float32) float32 {
	if max > value {
		return max
	}
	return value
}

// sigmoid returns activation for one input
func sigmoid(z float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-z))))
}

// sigmoidLayer applies sigmoid activation to each neuron's value, saving output
func sigmoidLayer(nn neuralNetwork, layer int) {
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = sigmoid(nn.layers[layer].neurons[neuron].value)
		nn.layers[layer].neurons[neuron].outputs = append(nn.layers[layer].neurons[neuron].outputs, nn.layers[layer].neurons[neuron].output)
	}
}

// sigmoidPrime returns sigmoid derivative
func sigmoidPrime(inputs [][]float32) [][]float32 {
	outputs := make([][]float32, len(inputs))
	for i, input := range inputs {
		outputs[i] = make([]float32, len(inputs[0]))
		for j, z := range input {
			outputs[i][j] = sigmoid(z) * (1 - sigmoid(z))
		}
	}
	return outputs
}

// softmax returns activation for a layer
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

// softmaxLayer applies softmax activation to a layer, saving output
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

// softmaxPrime returns softmax derivative
func softmaxPrime(z [][]float32) (d_Z [][]float32) {
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

// relu - Rectified Linear Unit, returns activation for one input
func relu(z float32) float32 {
	return max32(0, z)
}

// reluLayer applies relu activation to each neuron's value, saving output
func reluLayer(nn neuralNetwork, layer int) {
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = relu(nn.layers[layer].neurons[neuron].value)
		nn.layers[layer].neurons[neuron].outputs = append(nn.layers[layer].neurons[neuron].outputs, nn.layers[layer].neurons[neuron].output)
	}
}

// reluPrime - Rectified Linear Unit, returns activation derivative for one input
func reluPrime(z float32) float32 {
	if z > 0 {
		return 1
	}
	return 0
}

// reluPrime - Rectified Linear Unit, returns activation derivative for one layer
func reluPrimeLayer(inputs [][]float32) [][]float32 {
	outputs := make([][]float32, len(inputs))
	for i, input := range inputs {
		outputs[i] = make([]float32, len(inputs[0]))
		for j, z := range input {
			outputs[i][j] = reluPrime(z)
		}
	}
	return outputs
}

// leakyRelu - Leaky Rectified Linear Unit, returns activation for one input
func leakyRelu(z float32) float32 {
	var alpha float32 = 0.1
	if z >= 0 {
		return z
	}
	return z * alpha
}

// leakyRelu applies leakyRelu activation to each neuron's value, saving output
func leakyReluLayer(nn neuralNetwork, layer int) {
	for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
		nn.layers[layer].neurons[neuron].output = leakyRelu(nn.layers[layer].neurons[neuron].value)
		nn.layers[layer].neurons[neuron].outputs = append(nn.layers[layer].neurons[neuron].outputs, nn.layers[layer].neurons[neuron].output)
	}
}

// leakyReluPrime - Leaky Rectified Linear Unit, returns activation derivative for one input
func leakyReluPrime(z float32) float32 {
	var alpha float32 = 0.1
	if z >= 0 {
		return 1
	}
	return alpha
}

// leakyReluPrime - Leaky Rectified Linear Unit, returns activation derivative for one layer
func leakyReluPrimeLayer(inputs [][]float32) [][]float32 {
	outputs := make([][]float32, len(inputs))
	for i, input := range inputs {
		outputs[i] = make([]float32, len(inputs[0]))
		for j, z := range input {
			outputs[i][j] = leakyReluPrime(z)
		}
	}
	return outputs
}
