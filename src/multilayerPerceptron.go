package multilayer

import (
	"fmt"
	"math/rand"
	"os"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Printf("%vLaunching Multilayer Perceptron%v\n\n", BRIGHT, RESET)
	flagT, flagP, filepath := parseArg()
	rand.Seed(time.Now().UnixNano())

	data := preprocess()
	train_set, test_set := split(data)

	architecture := []int{len(data[0]) - 1, 16, 16, 16, 2}
	// architecture := []int{len(data[0]) - 1, 2, 2, 2} // test architecture ////
	nn := buildNN(architecture)

	_, err := os.Stat(filepath)
	if flagT || err != nil { // if model.json exists skip training, unless -t
		train(nn, train_set, test_set)
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
