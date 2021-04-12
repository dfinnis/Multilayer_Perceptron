package multilayer

import (
	"fmt"
	// "gonum/mat" // matrix linear algebra // gonum.org/v1/gonum/mat
)

type neuron struct {
	value float64
	bias float64
	weights []float64
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


func newNeuron(value float64) neuron {
	return neuron{
		value:				value,
	}
}

func newLayer(length uint8) layer {
	var neurons []neuron
	var i uint8
	for i = 0; i < length; i++ {
		neurons = append(neurons, newNeuron(42))
	}
	l := layer{length, neurons}
	return l
}


// MultilayerPerceptron is the main and only exposed function
func MultilayerPerceptron() {
	fmt.Println("Oh HI!!!!!!!!!!!!!!!!!!!!!!!!")

	nn := neuralNetwork{}
	nn.learningRate = 0.01

	// list := neuron{}
	// var hiddenNeurons []neuron
	// for i := 0; i < 16; i++ {
	// 	// hiddenNeurons = append(hiddenNeurons, neuron{42})
	// 	hiddenNeurons = append(hiddenNeurons, newNeuron(42))
	// }

	// var outputNeurons []neuron
	// for i := 0; i < 2; i++ {
	// 	outputNeurons = append(outputNeurons, newNeuron(1))
	// }

	// input := layer{16, hiddenNeurons}
	// hidden := layer{16, hiddenNeurons}
	// output := layer{2, outputNeurons}

	// nn.layers = append(nn.layers, input)
	// nn.layers = append(nn.layers, hidden)
	// nn.layers = append(nn.layers, hidden)
	// nn.layers = append(nn.layers, output)

	nn.layers = append(nn.layers, newLayer(16))
	nn.layers = append(nn.layers, newLayer(16))
	nn.layers = append(nn.layers, newLayer(16))
	nn.layers = append(nn.layers, newLayer(2))


	// fmt.Println(nn.config())
	fmt.Println(nn.learningRate)
	fmt.Println(nn.layers)
}
