package multilayer

import "fmt"

// predict returns predictions & ground truth for given samples
func predict(nn neuralNetwork, samples [][]float32) (predictions, y [][]float32) {
	input, y := splitXY(samples)
	predictions = feedforward(nn, input)
	return
}

// truthTally counts true & false, positives & negatives
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

// getMetrics converts true & false, positives & negatives into metrics
func getMetrics(tp, fn, fp, tn float32) (accuracy, precision, recall, specificity, F1_score float32) {
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	specificity = tn / (tn + fp)
	F1_score = (2 * (precision * recall)) / (precision + recall)
	if tp == 0 {
		precision = 0
		F1_score = 0
	}
	return
}

// predictFinal prints metrics for predictions on test set
func predictFinal(nn neuralNetwork, test_set [][]float32, modelPath string) {
	fmt.Printf("\n%v%vPredict%v\n\n", BOLD, UNDERLINE, RESET)
	loadModel(nn, modelPath)
	predictions, y := predict(nn, test_set)
	loss := binaryCrossEntropy(predictions, y)
	tp, fn, fp, tn := truthTally(predictions, y)
	printMetrics(tp, fn, fp, tn, loss)
	confusionMatrix(tp, fn, fp, tn)
}
