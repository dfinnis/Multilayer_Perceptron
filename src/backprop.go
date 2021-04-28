package multilayer

// getLayerValue finds the value for neurons in layer for all samples
func getLayerValue(nn neuralNetwork, output [][]float64, layer int) [][]float64 {
	var z [][]float64
	for sample, _ := range output {
		var layer_value []float64
		for _, neuron := range nn.layers[layer].neurons {
			layer_value = append(layer_value, neuron.z[sample])
		}
		z = append(z, layer_value)
	}
	return z
}

// getPrime finds the activation
func getPrime(nn neuralNetwork, layer int, z [][]float64) [][]float64 {
	prime := softmax_prime(z)
	if layer < len(nn.architecture)-1 {
		prime = sigmoid_prime(z)
	}
	return prime
}

// Activation derivative
func getActivationSlope(prime, d_A [][]float64) [][]float64 {
	var d_z [][]float64
	for i, sample := range prime {
		var layer_d []float64
		for j, value := range sample {
			layer_d = append(layer_d, d_A[i][j]*value)
		}
		d_z = append(d_z, layer_d)
	}
	return d_z
}

// getLayerOutput finds the previous layers outputs
func getLayerOutput(nn neuralNetwork, prime [][]float64, layer int) [][]float64 {
	var layerOutputs [][]float64
	for sample, _ := range prime {
		var layerOutput []float64
		for neuron := 0; neuron < nn.architecture[layer-1]; neuron++ {
			layerOutput = append(layerOutput, nn.layers[layer-1].neurons[neuron].outputs[sample])
		}
		layerOutputs = append(layerOutputs, layerOutput)
	}
	return layerOutputs
}

// getBiasSlope finds the bias derivative
func getBiasSlope(nn neuralNetwork, d_z [][]float64, layer int) []float64 {
	d_bias := make([]float64, nn.architecture[layer])
	for _, sample := range d_z {
		for neuron, _ := range nn.layers[layer].neurons {
			d_bias[neuron] += sample[neuron]
		}
	}
	return d_bias
}

// getLayerWeights finds all the layers weights
func getLayerWeights(nn neuralNetwork, layer int) [][]float64 {
	var weights [][]float64
	for _, neuron := range nn.layers[layer].neurons {
		var weightLayer []float64
		weightLayer = append(weightLayer, neuron.weights...)
		weights = append(weights, weightLayer)
	}
	return weights
}

// updateNN updates weights & bias with derivative of loss (SGD)
func updateNN(nn neuralNetwork, layer int, d_weights [][]float64, d_bias []float64, length float64) {
	for neuron, _ := range nn.layers[layer].neurons {
		for weight, _ := range nn.layers[layer].neurons[neuron].weights {
			nn.layers[layer].neurons[neuron].weights[weight] -= nn.learningRate * (d_weights[neuron][weight] / length)
		}
		nn.layers[layer].neurons[neuron].bias -= nn.learningRate * (d_bias[neuron] / length)
	}
}

// backpropLayer updates the layers weights and bias
func backpropLayer(nn neuralNetwork, output, y [][]float64, layer int, d_A [][]float64) [][]float64 {
	// fmt.Printf("layer: %v\n", layer) ////////////
	z := getLayerValue(nn, output, layer)
	prime := getPrime(nn, layer, z)
	d_z := getActivationSlope(prime, d_A)
	layerOutputs := getLayerOutput(nn, prime, layer)
	d_weights := multiply(transpose(d_z), layerOutputs)
	d_bias := getBiasSlope(nn, d_z, layer)
	weights := getLayerWeights(nn, layer)
	d_aLast := multiply(d_z, weights)
	updateNN(nn, layer, d_weights, d_bias, float64(len(output)))
	return d_aLast
}

// backprop updates network weights and bias from output layer down
func backprop(nn neuralNetwork, output, y [][]float64) {
	d_A := computeLossPrime(output, y)
	for layer := len(nn.architecture) - 1; layer > 0; layer-- {
		d_A = backpropLayer(nn, output, y, layer, d_A)
	}
}
