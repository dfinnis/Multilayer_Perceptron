package multilayer

import "fmt"

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Printf("\n%v%vLaunching Multilayer Perceptron%v\n\n", BOLD, UNDERLINE, RESET)

	// Flags
	flags := parseArg()

	// Data
	data := preprocess(flags.dataPath)
	train_set, test_set := splitData(data, flags)

	// Initialize
	nn := buildNN(len(data[0])-1, flags)

	// Train
	if flags.flagT || flags.err != nil { // if model.json exists skip training, unless -t
		train(nn, train_set, test_set, flags)
	}

	// Predict
	if flags.flagP || (!flags.flagT && !flags.flagP) {
		predictFinal(nn, test_set, flags.modelPath)
	}
}
