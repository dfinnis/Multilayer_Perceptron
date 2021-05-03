package multilayer

import (
	"fmt"
	"time"
)

// isNan returns true if given float32 is NaN
func isNaN(float float32) bool {
	return float != float
}

// train trains the network & saves the model
func train(nn neuralNetwork, train_set [][]float32, test_set [][]float32, flagE bool) {
	fmt.Printf("\n%v%vTrain model%v\n\n", BRIGHT, UNDERLINE, RESET)
	start := time.Now()

	for epoch := 1; epoch <= nn.epochs; epoch++ {
		input, y := splitXY(train_set)

		output := feedforward(nn, input)
		backprop(nn, output, y)

		trainLoss := nn.lossFunc(output, y)
		testLoss := predictLoss(nn, test_set)

		if isNaN(trainLoss) || (len(test_set) > 0 && isNaN(testLoss)) {
			break
		}
		// Early Stopping
		if epoch > 100 && flagE && testLoss > nn.testLoss[len(nn.testLoss)-1] {
			break
		}

		// Save Loss
		nn.trainLoss = append(nn.trainLoss, trainLoss)
		nn.testLoss = append(nn.testLoss, testLoss)

		// Print Metrics
		fmt.Printf("\rEpoch %5v/%v - Training loss: %-11v - Test loss: %-11v", epoch, nn.epochs, trainLoss, testLoss)

		saveModel(nn)
	}
	elapsed := time.Since(start)
	fmt.Printf("\n\nTraining time: %v\n\n", elapsed)
	fmt.Printf("Model saved as: model.json\n\n")
	visualize(nn.trainLoss, nn.testLoss)
}
