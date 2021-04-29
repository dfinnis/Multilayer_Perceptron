package multilayer

import (
	"fmt"
	"os"
)

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Printf("\n%v%vLaunching Multilayer Perceptron%v\n\n", BRIGHT, UNDERLINE, RESET)
	flagT, flagP, filepath, arch, flagE := parseArg()

	data := preprocess()
	train_set, test_set := split(data)

	var architecture []int
	architecture = append(architecture, len(data[0])-1)
	architecture = append(architecture, arch...)
	printArchitecture(architecture)

	nn := buildNN(architecture, flagE)

	_, err := os.Stat(filepath)
	if flagT || err != nil { // if model.json exists skip training, unless -t
		train(nn, train_set, test_set, flagE)
	} else {
		loadModel(nn, filepath)
	}

	if flagP {
		predictFinal(nn, test_set)
	}
}

// func dumpNN(nn neuralNetwork) {
// 	fmt.Println(nn.learningRate)
// 	fmt.Println()
// 	fmt.Println(nn.architecture)
// 	for i := 0; i < len(nn.architecture); i++ {
// 		fmt.Println()
// 		fmt.Println(nn.layers[i])
// 	}
// 	fmt.Println()
// }
