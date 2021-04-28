package multilayer

func backpropLayer(nn neuralNetwork, output, y [][]float64, layer int, d_A [][]float64) [][]float64 {
	// fmt.Printf("layer: %v\n", layer) ////////////
	var z [][]float64
	// Layer Value
	for sample, _ := range output {
		var layer_value []float64
		for _, neuron := range nn.layers[layer].neurons {
			layer_value = append(layer_value, neuron.z[sample])
		}
		z = append(z, layer_value)
	}
	// Activation
	prime := softmax_prime(z)
	if layer < len(nn.architecture)-1 {
		prime = sigmoid_prime(z)
	}

	// Activation derivative
	var d_z [][]float64
	for i, sample := range prime {
		var layer_d []float64
		for j, value := range sample {
			layer_d = append(layer_d, d_A[i][j]*value)
		}
		d_z = append(d_z, layer_d)
	}
	// Output
	var layerOutputs [][]float64
	for sample, _ := range prime {
		var layerOutput []float64
		for neuron := 0; neuron < nn.architecture[layer-1]; neuron++ {
			layerOutput = append(layerOutput, nn.layers[layer-1].neurons[neuron].outputs[sample])
		}
		layerOutputs = append(layerOutputs, layerOutput)
	}
	// Weights derivative
	d_weights := multiply(transpose(d_z), layerOutputs)
	// Bias derivative
	d_bias := make([]float64, nn.architecture[layer])
	for _, sample := range d_z {
		for neuron, _ := range nn.layers[layer].neurons {
			d_bias[neuron] += sample[neuron]
		}
	}
	// Activation of previous layer derivative
	var weights [][]float64
	for _, neuron := range nn.layers[layer].neurons {
		var weightLayer []float64
		for _, weight := range neuron.weights {
			weightLayer = append(weightLayer, weight)
		}
		weights = append(weights, weightLayer)
	}
	d_aLast := multiply(d_z, weights)
	// Update weights & bias with derivative of loss (SGD)
	for neuron, _ := range nn.layers[layer].neurons {
		for weight, _ := range nn.layers[layer].neurons[neuron].weights {
			nn.layers[layer].neurons[neuron].weights[weight] -= nn.learningRate * (d_weights[neuron][weight] / float64(len(output)))
		}
		nn.layers[layer].neurons[neuron].bias -= nn.learningRate * (d_bias[neuron] / float64(len(output)))
	}
	return d_aLast
}

func backprop(nn neuralNetwork, output, y [][]float64) {
	d_A := computeLossPrime(output, y)
	for layer := len(nn.architecture) - 1; layer > 0; layer-- {
		d_A = backpropLayer(nn, output, y, layer, d_A)
	}
}
