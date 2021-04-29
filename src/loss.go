package multilayer

import (
	"math"
)

// func meanSquaredError(outputs, y [][]float32) float32 {
// 	var loss float32
// 	for output := 0; output < len(outputs); output++ {
// 		diff := y[output][0] - outputs[output][0]
// 		loss += 2 * (diff * diff)
// 	}
// 	return loss / float32(len(outputs))
// }

// func rootMeanSquaredError(outputs, y [][]float32) float32 {
// 	var loss float32
// 	for output := 0; output < len(outputs); output++ {
// 		diff := y[output][0] - outputs[output][0]
// 		loss += 2 * (math.Sqrt(diff * diff))
// 	}
// 	return loss / float32(len(outputs))
// }

// Binary cross-entropy log loss
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

// Binary cross-entropy log loss
func computeLoss(outputs, y [][]float32) (lossSum float32) {
	return binaryCrossEntropy(outputs, y)
	// return meanSquaredError(outputs, y)
	// return rootMeanSquaredError(outputs, y)
}

// Binary cross-entropy log loss derivative
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
