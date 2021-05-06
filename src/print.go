package multilayer

import "fmt"

const RESET = "\x1B[0m"
const BOLD = "\x1B[1m"
const UNDERLINE = "\x1B[4m"
const RED = "\x1B[31m"
const GREEN = "\x1B[32m"

// printHeader prints intro
func printHeader(flags flags) {
	if !flags.flagQ {
		fmt.Printf("\033[H\033[2J") // Clear screen
	} else {
		fmt.Printf("\n")
	}
	fmt.Printf("%v%vLaunching Multilayer Perceptron%v\n\n", BOLD, UNDERLINE, RESET)
}

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

// printEpoch prints metrics each epoch
func printEpoch(epoch, epochs int, trainLoss, testLoss float32, flagQ bool, predictionsTrain, yTrain, predictionsTest, yTest [][]float32) {
	if flagQ {
		fmt.Printf("\rEpoch %5v/%v - Training loss: %-11v - Test loss: %-11v", epoch, epochs, trainLoss, testLoss)
	} else {
		tpTrain, fnTrain, fpTrain, tnTrain := truthTally(predictionsTrain, yTrain)
		accuracyTrain, precisionTrain, recallTrain, specificityTrain, F1_scoreTrain := getMetrics(tpTrain, fnTrain, fpTrain, tnTrain)
		tpTest, fnTest, fpTest, tnTest := truthTally(predictionsTest, yTest)
		accuracyTest, precisionTest, recallTest, specificityTest, F1_scoreTest := getMetrics(tpTest, fnTest, fpTest, tnTest)

		if epoch > 1 {
			fmt.Printf("\x1B[26F") // Cursor back to begining of print
		}
		fmt.Printf("+-------------------------------------------------+\n")
		fmt.Printf("|%v Epoch %5v%v/%-5v                               |\n", BOLD, epoch, RESET, epochs)
		fmt.Printf("+-----------------+---------------+---------------+\n")
		fmt.Printf("|%v Metric          %v|%v Training Set  %v|%v Test Set      %v|\n", BOLD, RESET, BOLD, RESET, BOLD, RESET)
		fmt.Printf("+-----------------+---------------+---------------+\n")
		fmt.Printf("|%v            Loss %v| %f      | %-8f      |\n", BOLD, RESET, trainLoss, testLoss)
		fmt.Printf("|                 |               |               |\n")
		fmt.Printf("|%v        Accuracy %v| %f      | %-8f      |\n", BOLD, RESET, accuracyTrain, accuracyTest)
		fmt.Printf("|                 |               |               |\n")
		fmt.Printf("|%v       Precision %v| %f      | %-8f      |\n", BOLD, RESET, precisionTrain, precisionTest)
		fmt.Printf("|                 |               |               |\n")
		fmt.Printf("|%v          Recall %v| %f      | %-8f      |\n", BOLD, RESET, recallTrain, recallTest)
		fmt.Printf("|                 |               |               |\n")
		fmt.Printf("|%v     Specificity %v| %f      | %-8f      |\n", BOLD, RESET, specificityTrain, specificityTest)
		fmt.Printf("|                 |               |               |\n")
		fmt.Printf("|%v        F1_score %v| %f      | %-8f      |\n", BOLD, RESET, F1_scoreTrain, F1_scoreTest)
		fmt.Printf("+-----------------+---------------+---------------+\n\n")

		confusionMatrix2(tpTrain, fnTrain, fpTrain, tnTrain, tpTest, fnTest, fpTest, tnTest)
	}
}

// confusionMatrix2 shows true & false, positives & negatives for training & test sets
func confusionMatrix2(tpTrain, fnTrain, fpTrain, tnTrain, tpTest, fnTest, fpTest, tnTest float32) {
	fmt.Printf("%vConfusion Matrix%v  +---------------+---------------+\n", BOLD, RESET)
	fmt.Printf("                  ǁ%v Ground Truth %v ǁ%v Ground Truth %v ǁ\n", BOLD, RESET, BOLD, RESET)
	fmt.Printf("                  ǁ-------+-------ǁ-------+-------ǁ\n")
	fmt.Printf("                  ǁ%v%v True %v |%v%v False %vǁ%v%v True %v |%v%v False %vǁ\n", BOLD, GREEN, RESET, BOLD, RED, RESET, BOLD, GREEN, RESET, BOLD, RED, RESET)
	fmt.Printf("+-----------------ǁ-------+-------ǁ-------+-------ǁ\n")
	fmt.Printf("|         |%v%v True %v ǁ%v %-5v %v|%v %-5v %vǁ%v %-5v %v|%v %-5v %vǁ\n", BOLD, GREEN, RESET, GREEN, tpTrain, RESET, RED, fpTrain, RESET, GREEN, tpTest, RESET, RED, fpTest, RESET)
	fmt.Printf("|%v Predict %v+-------ǁ-------+-------ǁ-------+-------ǁ\n", BOLD, RESET)
	fmt.Printf("|         |%v%v False %vǁ%v %-5v %v|%v %-5v %vǁ%v %-5v %v|%v %-5v %vǁ\n", BOLD, RED, RESET, RED, fnTrain, RESET, GREEN, tnTrain, RESET, RED, fnTest, RESET, GREEN, tnTest, RESET)
	fmt.Printf("+-----------------+-------+-------+-------+-------+")
}

// printMetrics shows the final metrics
func printMetrics(tp, fn, fp, tn, loss float32) {
	accuracy, precision, recall, specificity, F1_score := getMetrics(tp, fn, fp, tn)

	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n")
	fmt.Printf("|%v Metric      %v|%v Value    %v|%v Description                                                             %v|\n", BOLD, RESET, BOLD, RESET, BOLD, RESET)
	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n")
	fmt.Printf("|%v        Loss %v| %f | binary cross-entropy log loss                                           |\n", BOLD, RESET, loss)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v    Accuracy %v| %f | proportion of predictions classified correctly                          |\n", BOLD, RESET, accuracy)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v   Precision %v| %f | proportion of positive identifications correct                          |\n", BOLD, RESET, precision)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v      Recall %v| %f | proportion of actual positives correctly identified. True Positive Rate |\n", BOLD, RESET, recall)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v Specificity %v| %f | proportion of actual negatives correctly identified. True Negative Rate |\n", BOLD, RESET, specificity)
	fmt.Printf("|             |          |                                                                         |\n")
	fmt.Printf("|%v    F1_score %v| %f | harmonic mean of precision & recall. Max 1 (perfect), min 0             |\n", BOLD, RESET, F1_score)
	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n\n\n")

}

// confusionMatrix shows true & false, positives & negatives
func confusionMatrix(tp, fn, fp, tn float32) {
	fmt.Printf("%vConfusion Matrix%v  +---------------+\n", BOLD, RESET)
	fmt.Printf("                  ǁ%v Ground Truth %v ǁ\n", BOLD, RESET)
	fmt.Printf("Total: %-4v       ǁ-------+-------ǁ\n", (tp + fn + fp + tn))
	fmt.Printf("                  ǁ%v%v True %v |%v%v False %vǁ\n", BOLD, GREEN, RESET, BOLD, RED, RESET)
	fmt.Printf("+-----------------ǁ-------+-------ǁ\n")
	fmt.Printf("|         |%v%v True %v ǁ%v %-5v %v|%v %-5v %vǁ\n", BOLD, GREEN, RESET, GREEN, tp, RESET, RED, fp, RESET)
	fmt.Printf("|%v Predict %v+-------ǁ-------+-------ǁ\n", BOLD, RESET)
	fmt.Printf("|         |%v%v False %vǁ%v %-5v %v|%v %-5v %vǁ\n", BOLD, RED, RESET, RED, fn, RESET, GREEN, tn, RESET)
	fmt.Printf("+-----------------+-------+-------+\n\n")
}
