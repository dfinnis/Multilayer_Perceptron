package multilayer

import (
	"math"
)

func meanSquaredError(outputs, y [][]float64) float64 {
	var loss float64
	for output := 0; output < len(outputs); output++ {
		diff := y[output][0] - outputs[output][0]
		loss += 2 * (diff * diff)
	}
	return loss / float64(len(outputs))
}

func rootMeanSquaredError(outputs, y [][]float64) float64 {
	var loss float64
	for output := 0; output < len(outputs); output++ {
		diff := y[output][0] - outputs[output][0]
		loss += 2 * (math.Sqrt(diff * diff))
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

// Tested good!
func computeLossPrime(outputs [][]float64, y [][]float64) [][]float64 {
	var d_losses [][]float64
	for output := 0; output < len(outputs); output++ {
		loss := (y[output][0] / outputs[output][0]) - ((1 - y[output][0]) / (1 - outputs[output][0]))
		var d_loss []float64
		d_loss = append(d_loss, -loss)
		d_loss = append(d_loss, loss)

		d_losses = append(d_losses, d_loss)
	}
	// fmt.Printf("d_losses[0]: %v\n", d_losses[0]) ////////////
	return d_losses
}
