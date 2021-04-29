package multilayer

import "fmt"

func predict(nn neuralNetwork, samples [][]float64) (predictions, y [][]float64) {
	input, y := split_x_y(samples)
	predictions = feedforward(nn, input)
	return
}

func predictLoss(nn neuralNetwork, samples [][]float64) float64 {
	predictions, y := predict(nn, samples)
	loss := computeLoss(predictions, y)
	return loss
}

func truthTally(y_pred, y_true [][]float64) (float64, float64, float64, float64) {
	var tp float64 // True Positive		// Predicted True & Is True
	var fn float64 // False Negative	// Predicted False & Is True
	var fp float64 // False Positive	// Predicted True & Is False
	var tn float64 // True Negative		// Predicted False & Is False

	for i := 0; i < len(y_pred); i++ {
		if y_true[i][1] == 1 { // Is True
			if y_pred[i][1] > 0.5 { // Predicted True
				tp += 1
			} else { // Predicted False
				fn += 1
			}
		} else { // Is False
			if y_pred[i][1] > 0.5 { // Predicted True
				fp += 1
			} else { // Predicted False
				tn += 1
			}
		}
	}
	return tp, fn, fp, tn
}

func predictFinal(nn neuralNetwork, test_set [][]float64) {
	fmt.Printf("\n%v%vPredict%v\n\n", BRIGHT, UNDERLINE, RESET)
	predictions, y := predict(nn, test_set)
	loss := binaryCrossEntropy(predictions, y)
	tp, fn, fp, tn := truthTally(predictions, y)
	printMetrics(tp, fn, fp, tn, loss)
	confusionMatrix(tp, fn, fp, tn)
}
