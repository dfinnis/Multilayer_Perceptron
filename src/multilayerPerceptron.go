package multilayer

import (
	"fmt"
	"math/rand"
)

// seedRandom initializes rand with time or -s SEED
func seedRandom(flags flags) {
	rand.Seed(flags.seed)
	if !(flags.flagS || flags.flagQ) {
		fmt.Printf("Random seed: %d\n\n", flags.seed)
	}
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	flags := parseArg()
	printHeader(flags)
	seedRandom(flags)

	// Data
	data := preprocess(flags.dataPath)
	train_set, test_set := splitData(data, flags)

	// Initialize
	nn := buildNN(len(data[0])-1, flags)

	// Train
	if flags.flagT || (!flags.flagT && !flags.flagP) {
		train(nn, train_set, test_set, flags)
	}
	fmt.Printf("\x1B[?12;25h") // Reset Cursor

	// Predict
	if flags.flagP {
		predictFinal(nn, test_set, flags.modelPath)
	}
}
