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
func train(nn neuralNetwork, train_set [][]float32, test_set [][]float32, flags flags) {
	fmt.Printf("\n%v%vTrain model%v\n\n", BOLD, UNDERLINE, RESET)
	start := time.Now()

	for epoch := 1; epoch <= nn.epochs; epoch++ {
		input, yTrain := splitXY(train_set)

		// Train
		predictionsTrain := feedforward(nn, input)
		backprop(nn, predictionsTrain, yTrain)

		// Loss
		trainLoss := nn.lossFunc(predictionsTrain, yTrain)
		predictionsTest, yTest := predict(nn, test_set)
		testLoss := nn.lossFunc(predictionsTest, yTest)

		if isNaN(trainLoss) || (len(test_set) > 0 && isNaN(testLoss)) {
			break
		}

		// Early Stopping
		if flags.flagE && epoch > 100 && testLoss > nn.testLoss[len(nn.testLoss)-1] {
			break
		}

		printEpoch(epoch, nn.epochs, trainLoss, testLoss, flags.flagQ, predictionsTrain, yTrain, predictionsTest, yTest)
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
