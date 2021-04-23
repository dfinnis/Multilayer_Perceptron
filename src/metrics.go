package multilayer

import "fmt"

const RESET = "\x1B[0m"
const BRIGHT = "\x1B[1m"
const RED = "\x1B[31m"
const GREEN = "\x1B[32m"

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

	fmt.Printf("                     +---------------+\n")
	fmt.Printf("                     |%v Ground Truth %v |\n", BRIGHT, RESET)
	fmt.Printf("                     +-------+-------+\n")
	fmt.Printf("                     |%v%v True %v |%v%v False %v|\n", BRIGHT, GREEN, RESET, BRIGHT, RED, RESET)
	fmt.Printf("+--------------------+-------+-------+\n")
	fmt.Printf("|            |%v%v True %v |%v %-5v %v|%v %-5v %v|\n", BRIGHT, GREEN, RESET, GREEN, tp, RESET, RED, fp, RESET)
	fmt.Printf("|%v Prediction %v+-------+-------+-------+\n", BRIGHT, RESET)
	fmt.Printf("|            |%v%v False %v|%v %-5v %v|%v %-5v %v|\n", BRIGHT, RED, RESET, RED, fn, RESET, GREEN, tn, RESET)
	fmt.Printf("+--------------------+-------+-------+\n\n")

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
