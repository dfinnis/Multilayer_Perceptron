package multilayer

import (
	"math"
)

func meanSquaredError(outputs, y [][]float64) float64 {
	var loss float64
	for output := 0; output < len(outputs); output++ {
		diff := y[output][0] - outputs[output][0]
		loss += diff * diff
	}
	return loss / float64(len(outputs))
}

func rootMeanSquaredError(outputs, y [][]float64) float64 {
	var loss float64
	for output := 0; output < len(outputs); output++ {
		diff := y[output][0] - outputs[output][0]
		loss += math.Sqrt(diff * diff)
	}
	return loss / float64(len(outputs))
}

// Binary cross-entropy log loss
func binaryCrossEntropy(outputs, y [][]float64) (lossSum float64) {
	// fmt.Printf("\nlen(outputs): %v\n", len(outputs)) ////////////
	// fmt.Printf("len(outputs[0]): %v\n", len(outputs[0])) ////////////
	// fmt.Printf("len(y): %v\n", len(y)) ////////////
	// fmt.Printf("len(y[0]): %v\n", len(y[0])) ////////////
	// var loss float64 /////////////
	var loss float64
	for output := 0; output < len(outputs); output++ {
		for dignosis := 0; dignosis < len(outputs[0]); dignosis++ {
			// fmt.Printf("lets go: %v\n", (1 - outputs[output][dignosis])) ////////////
			// fmt.Printf("Log: %v\n", math.Log(1 - outputs[output][dignosis])) ////////////
			// fmt.Printf("y[output]: %v\n", y[output][dignosis]) ////////////
			loss += y[output][dignosis]*math.Log(outputs[output][dignosis]) + (1-y[output][dignosis])*math.Log(1-outputs[output][dignosis])
			// fmt.Printf("loss here: %v\n", y[output][dignosis] * math.Log(outputs[output][dignosis]) + (1 - y[output][dignosis]) * math.Log(1 - outputs[output][dignosis])) ////////////
		}
		// break ///////
	}
	lossSum = -1 / float64(len(outputs)) * loss
	return
}

// Binary cross-entropy log loss
func computeLoss(outputs, y [][]float64) (lossSum float64) {
	// return meanSquaredError(outputs, y)
	// return rootMeanSquaredError(outputs, y)
	return binaryCrossEntropy(outputs, y)
}

func computeLossPrime(outputs [][]float64, y [][]float64) (d_losses [][]float64) {
	for output := 0; output < len(outputs); output++ {
		var d_loss []float64
		for diagnosis := 0; diagnosis <= 1; diagnosis++ {
			d_loss = append(d_loss, -(y[output][diagnosis]/outputs[output][diagnosis])-((1-y[output][diagnosis])/(1-outputs[output][diagnosis])))
		}
		d_losses = append(d_losses, d_loss)
	}
	return
}

// func computeLossPrime(outputs [][]float64, y [][]float64) (d_losses [][]float64) {
// 	for output := 0; output < len(outputs); output++ {
// 		var b []float64
// 		b = append(b, y[output][0] / outputs[output][0])
// 		b = append(b, y[output][1] / outputs[output][1])

// 		var m []float64
// 		m = append(m, (1 - y[output][0]) / (1 - outputs[output][0]))
// 		m = append(m, (1 - y[output][1]) / (1 - outputs[output][1]))

// 		var d_loss []float64
// 		d_loss = append(d_loss, - (b[0] - m[0]))
// 		d_loss = append(d_loss, - (b[1] - m[1]))
// 		d_losses = append(d_losses, d_loss)
// 	}
// 	return
// }

func backpropLayer(nn neuralNetwork, output, y [][]float64, layer int, d_A [][]float64) [][]float64 {
	// fmt.Printf("layer: %v\n", layer) ////////////
	var z [][]float64
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

	var d_z [][]float64
	for i, sample := range prime {
		var layer_d []float64
		for j, value := range sample {
			layer_d = append(layer_d, d_A[i][j]*value)
		}
		d_z = append(d_z, layer_d)
	}

	var layerOutputs [][]float64
	for sample, _ := range prime {
		var layerOutput []float64
		for neuron := 0; neuron < nn.architecture[layer-1]; neuron++ {
			layerOutput = append(layerOutput, nn.layers[layer-1].neurons[neuron].outputs[sample])
		}
		layerOutputs = append(layerOutputs, layerOutput)
	}

	d_weights := multiply(transpose(d_z), layerOutputs)

	d_bias := make([]float64, nn.architecture[layer])
	for _, sample := range d_z {
		for neuron, _ := range nn.layers[layer].neurons {
			d_bias[neuron] += sample[neuron]
		}
	}

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

	// update weights & bias with derivative of loss (SGD)
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
