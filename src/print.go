package multilayer

import "fmt"

const RESET = "\x1B[0m"
const BOLD = "\x1B[1m"
const UNDERLINE = "\x1B[4m"
const RED = "\x1B[31m"
const GREEN = "\x1B[32m"

// printSplit shows how the data is split between training & test set
func printSplit(train_set, test_set int) {
	fmt.Printf("+--------------+---------+\n")
	fmt.Printf("|%v Data Split   %v|%v Samples %v|\n", BOLD, RESET, BOLD, RESET)
	fmt.Printf("+--------------+---------+\n")
	fmt.Printf("| Training set | %-7v |\n", train_set)
	fmt.Printf("| Test set     | %-7v |\n", test_set)
	fmt.Printf("+--------------+---------+\n\n")
}

// printArchitecture shows the network shape
func printArchitecture(architecture []int) {
	fmt.Printf("+----------------------------------+\n")
	fmt.Printf("|%v   Neural Network Architecture %v   |\n", BOLD, RESET)
	fmt.Printf("+-----------+---------+------------+\n")
	fmt.Printf("|%v Layer     %v|%v Neurons %v|%v Activation %v|\n", BOLD, RESET, BOLD, RESET, BOLD, RESET)
	fmt.Printf("+-----------+---------+------------+\n")
	for i, layer := range architecture {
		label := "Hidden"
		activation := "Sigmoid"
		if i == 0 {
			label = "Input"
			activation = "None"
		} else if i == len(architecture)-1 {
			label = "Output"
			activation = "Softmax"
		}
		fmt.Printf("| %-2v %-6v | %-7v | %-10v |\n", i+1, label, layer, activation)
	}
	fmt.Printf("+-----------+---------+------------+\n\n")
}

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

// printMetrics shows the final metrics
func printMetrics(tp, fn, fp, tn, loss float32) {
	accuracy, precision, recall, specificity, F1_score := getMetrics(tp, fn, fp, tn)

	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n")
	fmt.Printf("|%v Metric      %v|%v Value    %v|%v Description                                                             %v|\n", BOLD, RESET, BOLD, RESET, BOLD, RESET)
	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n")
	fmt.Printf("|%v Accuracy    %v| %f | proportion of predictions classified correctly                          |\n", BOLD, RESET, accuracy)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v Precision   %v| %f | proportion of positive identifications correct                          |\n", BOLD, RESET, precision)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v Recall      %v| %f | proportion of actual positives identified correctly. True Positive Rate |\n", BOLD, RESET, recall)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v Specificity %v| %f | proportion of actual negatives identified correctly. True Negative Rate |\n", BOLD, RESET, specificity)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v F1_score    %v| %f | harmonic mean of precision and recall. Max 1 (perfect), min 0           |\n", BOLD, RESET, F1_score)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v Loss        %v| %f | binary cross-entropy log loss                                           |\n", BOLD, RESET, loss)
	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n\n\n")

}

// confusionMatrix shows true & false, positives & negatives
func confusionMatrix(tp, fn, fp, tn float32) {
	fmt.Printf("%vConfusion Matrix%v     +---------------+\n", BOLD, RESET)
	fmt.Printf("                     |%v Ground Truth %v |\n", BOLD, RESET)
	fmt.Printf("Total samples: %-5v +-------+-------+\n", (tp + fn + fp + tn))
	fmt.Printf("                     |%v%v True %v |%v%v False %v|\n", BOLD, GREEN, RESET, BOLD, RED, RESET)
	fmt.Printf("+--------------------+-------+-------+\n")
	fmt.Printf("|            |%v%v True %v |%v %-5v %v|%v %-5v %v|\n", BOLD, GREEN, RESET, GREEN, tp, RESET, RED, fp, RESET)
	fmt.Printf("|%v Prediction %v+-------+-------+-------+\n", BOLD, RESET)
	fmt.Printf("|            |%v%v False %v|%v %-5v %v|%v %-5v %v|\n", BOLD, RED, RESET, RED, fn, RESET, GREEN, tn, RESET)
	fmt.Printf("+--------------------+-------+-------+\n\n")
}
