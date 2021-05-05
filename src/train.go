package multilayer

import (
	"fmt"
	"time"
)

// isNan returns true if given float32 is NaN
func isNaN(float float32) bool {
	return float != float
}

// printEpoch prints metrics each epoch
func printEpoch(epoch, epochs int, trainLoss, testLoss float32, flagQ bool, output, y [][]float32) {
	// if flagQ {
	fmt.Printf("\rEpoch %5v/%v - Training loss: %-11v - Test loss: %-11v", epoch, epochs, trainLoss, testLoss)
	// } else {
	// 	tp, fn, fp, tn := truthTally(output, y)
	// 	accuracy, precision, recall, specificity, F1_score := getMetrics(tp, fn, fp, tn)

	// 	fmt.Printf("+--------------------+---------------+-------------------------------------------------------------+\n")
	// 	fmt.Printf("|%v Epoch %5v%v/%-6v |%v Training Set  %v|%v Test Set %v|\n", BOLD, epoch, RESET, epochs, BOLD, RESET, BOLD, RESET)
	// 	fmt.Printf("+--------------------+---------------+-------------------------------------------------------------+\n")
	// 	// fmt.Printf("|%v Loss        %v| %f | binary cross-entropy log loss                                           |\n", BOLD, RESET, loss)
	// 	fmt.Printf("|             |          |                                                                         |\n")
	// 	fmt.Printf("|%v Accuracy    %v| %f | proportion of predictions classified correctly                          |\n", BOLD, RESET, accuracy)
	// 	fmt.Printf("|             |          |                                                                         |\n")
	// 	fmt.Printf("|%v Precision   %v| %f | proportion of positive identifications correct                          |\n", BOLD, RESET, precision)
	// 	fmt.Printf("|             |          |                                                                         |\n")
	// 	fmt.Printf("|%v Recall      %v| %f | proportion of actual positives identified correctly. True Positive Rate |\n", BOLD, RESET, recall)
	// 	fmt.Printf("|             |          |                                                                         |\n")
	// 	fmt.Printf("|%v Specificity %v| %f | proportion of actual negatives identified correctly. True Negative Rate |\n", BOLD, RESET, specificity)
	// 	fmt.Printf("|             |          |                                                                         |\n")
	// 	fmt.Printf("|%v F1_score    %v| %f | harmonic mean of precision and recall. Max 1 (perfect), min 0           |\n", BOLD, RESET, F1_score)
	// 	fmt.Printf("+-------------+----------+-------------------------------------------------------------------------+\n\n\n")

	// 	// printMetrics(tp, fn, fp, tn, loss)
	// 	confusionMatrix(tp, fn, fp, tn)
	// 	fmt.Printf("\x1B[13F") // Cursor back to begining of previous line
	// }
}

// train trains the network & saves the model
func train(nn neuralNetwork, train_set [][]float32, test_set [][]float32, flags flags) {
	fmt.Printf("\n%v%vTrain model%v\n\n", BOLD, UNDERLINE, RESET)
	start := time.Now()

	for epoch := 1; epoch <= nn.epochs; epoch++ {
		input, y := splitXY(train_set)

		// Train
		output := feedforward(nn, input)
		backprop(nn, output, y)

		// Loss
		trainLoss := nn.lossFunc(output, y)
		testLoss := predictLoss(nn, test_set)
		if isNaN(trainLoss) || (len(test_set) > 0 && isNaN(testLoss)) {
			break
		}

		// Early Stopping
		if flags.flagE && epoch > 100 && testLoss > nn.testLoss[len(nn.testLoss)-1] {
			break
		}

		printEpoch(epoch, nn.epochs, trainLoss, testLoss, flags.flagQ, output, y)
		// Save
		nn.trainLoss = append(nn.trainLoss, trainLoss)
		nn.testLoss = append(nn.testLoss, testLoss)
		saveModel(nn)
	}
	elapsed := time.Since(start)
	fmt.Printf("\n\nTraining time: %v\n\n", elapsed)
	fmt.Printf("Model saved as: model.json\n\n")
	visualize(nn.trainLoss, nn.testLoss)
}
