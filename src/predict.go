package multilayer

import "fmt"

func predict(nn neuralNetwork, samples [][]float32) (predictions, y [][]float32) {
	input, y := splitXY(samples)
	predictions = feedforward(nn, input)
	return
}

func predictLoss(nn neuralNetwork, samples [][]float32) float32 {
	predictions, y := predict(nn, samples)
	loss := computeLoss(predictions, y)
	return loss
}

func truthTally(y_pred, y_true [][]float32) (float32, float32, float32, float32) {
	var tp float32 // True Positive		// Predicted True & Is True
	var fn float32 // False Negative	// Predicted False & Is True
	var fp float32 // False Positive	// Predicted True & Is False
	var tn float32 // True Negative		// Predicted False & Is False

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

func predictFinal(nn neuralNetwork, test_set [][]float32) {
	fmt.Printf("\n%v%vPredict%v\n\n", BRIGHT, UNDERLINE, RESET)
	predictions, y := predict(nn, test_set)
	loss := binaryCrossEntropy(predictions, y)
	tp, fn, fp, tn := truthTally(predictions, y)
	printMetrics(tp, fn, fp, tn, loss)
	confusionMatrix(tp, fn, fp, tn)
}
