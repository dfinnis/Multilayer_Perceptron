package multilayer

import (
	"fmt"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

type neuron struct {
	value float64
	// weights []float64
	// bias float64
}

type layer struct {
	length uint8
	// neurons []neuron
	// activation function_pointer (sigmoid / softmax)
}

type neuralNetwork struct {
	learningRate float64
	layers []layer
}

// func (nn neuralNetwork) config() (float64, int) {
// 	return nn.learningRate, 1
// }

// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Println("Oh HI!!!!!!!!!!!!!!!!!!!!!!!!")

	nn := neuralNetwork{}
	nn.learningRate = 0.01

	input := layer{16}
	hidden := layer{16}
	output := layer{2}

	nn.layers = append(nn.layers, input)
	nn.layers = append(nn.layers, hidden)
	nn.layers = append(nn.layers, hidden)
	nn.layers = append(nn.layers, output)

	// fmt.Println(nn.config())
	fmt.Println(nn.learningRate)
	fmt.Println(nn.layers)
}
