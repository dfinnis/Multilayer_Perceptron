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
	// d_A[0][0] = 1                      ////////////
	// d_A[0][1] = -1                     ////////////
	// z[0][0] = 2                        ////////////
	// z[0][1] = 3                        ////////////
	// fmt.Printf("d_A[0]: %v\n", d_A[0]) ////////////
	// fmt.Printf("z[0]: %v\n", z[0])     ////////////
	// Activation
	prime := softmax_prime(z)
	if layer < len(nn.architecture)-1 {
		prime = sigmoid_prime(z)
	}
	// fmt.Printf("prime[0]: %v\n", prime[0]) ////////////

	// Activation derivative
	var d_z [][]float64
	for i, sample := range prime {
		var layer_d []float64
		for j, value := range sample {
			layer_d = append(layer_d, d_A[i][j]*value)
		}
		d_z = append(d_z, layer_d)
	}
	// fmt.Printf("d_z[0]: %v\n", d_z[0]) ////////////
	// Output
	var layerOutputs [][]float64
	for sample, _ := range prime {
		var layerOutput []float64
		for neuron := 0; neuron < nn.architecture[layer-1]; neuron++ {
			layerOutput = append(layerOutput, nn.layers[layer-1].neurons[neuron].outputs[sample])
		}
		layerOutputs = append(layerOutputs, layerOutput)
	}
	// fmt.Printf("layerOutputs[0]: %v\n", layerOutputs[0])          ////////////
	// fmt.Printf("len(layerOutputs: %v\n", len(layerOutputs))       ////////////
	// fmt.Printf("len(layerOutputs[0]: %v\n", len(layerOutputs[0])) ////////////
	// fmt.Printf("len(z: %v\n", len(z))                             ////////////
	// fmt.Printf("len(z[0]: %v\n", len(z[0]))                       ///////////

	// Weights derivative
	// a := mat.NewDense(len(d_z), len(d_z[0]), d_z[0])
	// for i, _ := range d_z {
	// 	for j, _ := range d_z[i] {
	// 		&a[i][j] = d_z[i][j]
	// 	}
	// }
	// var d_weights_test mat.Dense
	// d_weights_test.Mul(d_z, transpose(layerOutputs))

	// Print the result using the formatter.
	// fc := mat.Formatted(a, mat.Prefix("    "), mat.Squeeze())
	// fmt.Printf("c = %v", fc)

	d_weights := multiply(transpose(d_z), layerOutputs) // correct?
	// d_weights := dotProduct(layerOutputs, transpose(d_z)) // correct?
	// if layer == 3 {
	// 	fmt.Printf("d_weights: %v\n", d_weights) //////////
	// }
	// d_weights[0][0] = 1                                 /////////////////////
	// var flip_weights [][]float64                      ////////////////////////
	// for neuron := len(d_weights) - 1; neuron >= 0; neuron--{
	// 	flip_weights = append(flip_weights, d_weights[neuron]) ////////////////////////
	// }

	// fmt.Printf("len(d_weights: %v\n", len(d_weights))       ////////////
	// fmt.Printf("len(d_weights[0]: %v\n", len(d_weights[0])) ///////////

	// fmt.Printf("d_weights[0]: %v\n", d_weights[0])          ///////////
	// fmt.Printf("d_weights[1]: %v\n", d_weights[1])          ///////////
	// fmt.Printf("flip_weights: %v\n", flip_weights)          ///////////

	// os.Exit(1) //////////////
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
	d_aLast := multiply2(d_z, weights)

	// fmt.Printf("d_bias): %v\n", d_bias) ////////////
	// fmt.Printf("d_weights): %v\n", d_weights) ////////////
	// fmt.Printf("len(d_bias): %v\n", len(d_bias)) ////////////
	// fmt.Printf("shape(d_weights): %v %v\n", len(d_weights), len(d_weights[0])) ////////////

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
