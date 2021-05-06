package multilayer

// getLayerValue returns the value for neurons in layer for all samples
func getLayerValue(nn neuralNetwork, output [][]float32, layer int) [][]float32 {
	var z [][]float32
	for sample, _ := range output {
		var layer_value []float32
		for _, neuron := range nn.layers[layer].neurons {
			layer_value = append(layer_value, neuron.z[sample])
		}
		z = append(z, layer_value)
	}
	return z
}

// getPrime returns the activation
func getPrime(nn neuralNetwork, layer int, z [][]float32) [][]float32 {
	prime := softmaxPrime(z)
	if layer < len(nn.architecture)-1 {
		prime = sigmoidPrimeLayer(z)
	}
	return prime
}

// getActivationSlope returns the activation derivative
func getActivationSlope(prime, d_A [][]float32) [][]float32 {
	var d_z [][]float32
	for i, sample := range prime {
		var layer_d []float32
		for j, value := range sample {
			layer_d = append(layer_d, d_A[i][j]*value)
		}
		d_z = append(d_z, layer_d)
	}
	return d_z
}

// getLayerOutput returns the previous layers outputs
func getLayerOutput(nn neuralNetwork, prime [][]float32, layer int) [][]float32 {
	var layerOutputs [][]float32
	for sample, _ := range prime {
		var layerOutput []float32
		for neuron := 0; neuron < nn.architecture[layer-1]; neuron++ {
			layerOutput = append(layerOutput, nn.layers[layer-1].neurons[neuron].outputs[sample])
		}
		layerOutputs = append(layerOutputs, layerOutput)
	}
	return layerOutputs
}

// getBiasSlope returns the bias derivative
func getBiasSlope(nn neuralNetwork, d_z [][]float32, layer int) []float32 {
	d_bias := make([]float32, nn.architecture[layer])
	for _, sample := range d_z {
		for neuron, _ := range nn.layers[layer].neurons {
			d_bias[neuron] += sample[neuron]
		}
	}
	return d_bias
}

// getLayerWeights returns all the layers weights
func getLayerWeights(nn neuralNetwork, layer int) [][]float32 {
	var weights [][]float32
	for _, neuron := range nn.layers[layer].neurons {
		var weightLayer []float32
		weightLayer = append(weightLayer, neuron.weights...)
		weights = append(weights, weightLayer)
	}
	return weights
}

// updateNN updates weights & bias with derivative of loss (SGD)
func updateNN(nn neuralNetwork, layer int, d_weights [][]float32, d_bias []float32, length float32) {
	for neuron, _ := range nn.layers[layer].neurons {
		for weight, _ := range nn.layers[layer].neurons[neuron].weights {
			nn.layers[layer].neurons[neuron].weights[weight] -= nn.learningRate * (d_weights[neuron][weight] / length)
		}
		nn.layers[layer].neurons[neuron].bias -= nn.learningRate * (d_bias[neuron] / length)
	}
}

// backpropLayer updates the layers weights & bias
func backpropLayer(nn neuralNetwork, output, y [][]float32, layer int, d_A [][]float32) [][]float32 {
	z := getLayerValue(nn, output, layer)
	prime := getPrime(nn, layer, z)
	d_z := getActivationSlope(prime, d_A)
	layerOutputs := getLayerOutput(nn, prime, layer)

	// Weights & bias derivative
	d_weights := multiply(transpose(d_z), layerOutputs)
	d_bias := getBiasSlope(nn, d_z, layer)

	// Previous layer derivative
	weights := getLayerWeights(nn, layer)
	d_aLast := multiply(d_z, weights)

	// Update weights & bias
	updateNN(nn, layer, d_weights, d_bias, float32(len(output)))
	return d_aLast
}

// backprop updates network weights and bias from output layer down
func backprop(nn neuralNetwork, output, y [][]float32) {
	d_A := computeLossPrime(output, y)
	for layer := len(nn.architecture) - 1; layer > 0; layer-- {
		d_A = backpropLayer(nn, output, y, layer, d_A)
	}
}
