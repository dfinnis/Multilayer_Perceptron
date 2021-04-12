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
	neurons []neuron
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

	// n := neuron{42} // init random

	// list := neuron{}
	var hiddenNeurons []neuron
	for i := 0; i < 16; i++ {
		hiddenNeurons = append(hiddenNeurons, neuron{42})
	}

	var outputNeurons []neuron
	for i := 0; i < 2; i++ {
		outputNeurons = append(outputNeurons, neuron{1})
	}

	input := layer{16, hiddenNeurons}
	hidden := layer{16, hiddenNeurons}
	output := layer{2, outputNeurons}

	nn.layers = append(nn.layers, input)
	nn.layers = append(nn.layers, hidden)
	nn.layers = append(nn.layers, hidden)
	nn.layers = append(nn.layers, output)

	// fmt.Println(nn.config())
	fmt.Println(nn.learningRate)
	fmt.Println(nn.layers)
}
