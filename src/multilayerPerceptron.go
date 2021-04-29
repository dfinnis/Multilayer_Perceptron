package multilayer

import (
	"fmt"
)

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Printf("\n%v%vLaunching Multilayer Perceptron%v\n\n", BRIGHT, UNDERLINE, RESET)
	flagT, dataPath, flagP, modelPath, architecture, flagE, flagS, err := parseArg()

	data := preprocess(dataPath)
	train_set, test_set := splitData(data, flagT, flagP, flagS, err)

	nn := buildNN(len(data[0])-1, architecture, flagE)

	if flagT || err != nil { // if model.json exists skip training, unless -t
		train(nn, train_set, test_set, flagE)
	} else {
		loadModel(nn, modelPath)
	}

	if flagP || (!flagT && !flagP) {
		predictFinal(nn, test_set)
	}
}
