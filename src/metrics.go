package multilayer

import "fmt"

func truthTeller(y_pred, y_true [][]float64) {
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

	fmt.Printf("tp %v\n", tp)   ////////////////
	fmt.Printf("fn %v\n", fn)   ////////////////
	fmt.Printf("fp %v\n", fp)   ////////////////
	fmt.Printf("tn %v\n\n", tn) ////////////////

	accuracy := (tp + tn) / (tp + tn + fp + fn)
	precision := tp / (tp + fp)
	recall := tp / (tp + fn)
	specificity := tn / (tn + fp)
	F1_score := (2 * (precision * recall)) / (precision + recall)

	fmt.Printf("Accuracy: %v\n\n", accuracy)
	fmt.Printf("Precision: %v\n\n", precision)
	fmt.Printf("Recall: %v\n\n", recall)
	fmt.Printf("Specificity: %v\n\n", specificity)
	fmt.Printf("F1_score: %v\n\n", F1_score)
}

func metrics(nn neuralNetwork, test_set [][]float64) {
	predictions, y := predict(nn, test_set)
	loss := computeLoss(predictions, y)
	fmt.Printf("Final loss on validation set: %v\n\n", loss)
	truthTeller(predictions, y)
}
