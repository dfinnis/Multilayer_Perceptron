package multilayer

import "fmt"

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Printf("\n%v%vLaunching Multilayer Perceptron%v\n\n", BRIGHT, UNDERLINE, RESET)

	// Flags
	flagT, dataPath, flagP, modelPath, architecture, flagE, flagS, flagMSE, flagRMSE, learningRate, epochs, err := parseArg()

	// Data
	data := preprocess(dataPath)
	train_set, test_set := splitData(data, flagT, flagP, flagS, err)

	// Initialize
	nn := buildNN(len(data[0])-1, architecture, flagE, flagMSE, flagRMSE, learningRate, epochs)

	// Train
	if flagT || err != nil { // if model.json exists skip training, unless -t
		train(nn, train_set, test_set, flagE)
	} else {
		loadModel(nn, modelPath)
	}

	// Predict
	if flagP || (!flagT && !flagP) {
		predictFinal(nn, test_set)
	}
}
