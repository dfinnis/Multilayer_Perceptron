package multilayer

import (
	"fmt"
	"math/rand"
	"time"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

func predict(nn neuralNetwork, samples [][]float64) (predictions, y [][]float64) {
	input, y := split_x_y(samples)
	predictions = feedforward(nn, input)
	return
}

func predictLoss(nn neuralNetwork, samples [][]float64) float64 {
	predictions, y := predict(nn, samples)
	loss := computeLoss(predictions, y)
	return loss
}

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Printf("%vLaunching Multilayer Perceptron...%v\n\n", "\x1b[1m", "\x1b[0m")
	rand.Seed(time.Now().UnixNano())

	data := preprocess()
	train_set, test_set := split(data)

	architecture := []int{len(data[0]) - 1, 16, 16, 16, 2}
	// architecture := []int{len(data[0]) - 1, 2, 2, 2} // test architecture ////
	nn := buildNN(architecture)

	train(nn, train_set, test_set)

	metrics(nn, test_set)
	// loadModel(nn)
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
