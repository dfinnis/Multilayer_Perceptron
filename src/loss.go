package multilayer

import (
	"math"
)

// Mean Squared Error loss
func meanSquaredError(outputs, y [][]float32) float32 {
	var loss float32
	for output := 0; output < len(outputs); output++ {
		diff := y[output][0] - outputs[output][0]
		loss += 2 * (diff * diff)
	}
	return loss / float32(len(outputs))
}

// Root Mean Squared Error loss
func rootMeanSquaredError(outputs, y [][]float32) float32 {
	var loss float32
	for output := 0; output < len(outputs); output++ {
		diff := y[output][0] - outputs[output][0]
		loss += 2 * float32(math.Sqrt(float64(diff*diff)))
	}
	return loss / float32(len(outputs))
}

// Binary Cross-Entropy log loss
func binaryCrossEntropy(outputs, y [][]float32) (lossSum float32) {
	var loss float32
	for output := 0; output < len(outputs); output++ {
		for dignosis := 0; dignosis < len(outputs[0]); dignosis++ {
			loss += y[output][dignosis]*float32(math.Log(float64(outputs[output][dignosis]))) + (1-y[output][dignosis])*float32(math.Log(float64(1-outputs[output][dignosis])))
		}
	}
	lossSum = -1 / float32(len(outputs)) * loss
	return
}

// Binary Cross-Entropy log loss derivative
func computeLossPrime(outputs [][]float32, y [][]float32) [][]float32 {
	var d_losses [][]float32
	for output := 0; output < len(outputs); output++ {
		loss := (y[output][0] / outputs[output][0]) - ((1 - y[output][0]) / (1 - outputs[output][0]))
		var d_loss []float32
		d_loss = append(d_loss, -loss)
		d_loss = append(d_loss, loss)
		d_losses = append(d_losses, d_loss)
	}
	return d_losses
}
