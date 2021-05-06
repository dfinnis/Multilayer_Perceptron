package multilayer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

// Save/load model, 2D array [][]jsonNeuron <=> json file
type jsonNeuron struct {
	Bias    float32
	Weights []float32
}

// Save bias & weights to model.json
func saveModel(nn neuralNetwork) {
	model := make([][]jsonNeuron, len(nn.architecture)-1)
	for layer := 1; layer < len(nn.architecture); layer++ {
		neurons := make([]jsonNeuron, nn.architecture[layer])
		model[layer-1] = neurons
		for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
			model[layer-1][neuron].Bias = nn.layers[layer].neurons[neuron].bias
			for weight := 0; weight < len(nn.layers[layer].neurons[neuron].weights); weight++ {
				model[layer-1][neuron].Weights = append(model[layer-1][neuron].Weights, nn.layers[layer].neurons[neuron].weights[weight])
			}
		}
	}
	jsonString, err := json.MarshalIndent(model, "", "	")
	checkError("json.Marshal", err)
	err = ioutil.WriteFile("model.json", jsonString, 0644)
	checkError("ioutil.WriteFile", err)
}

// isJSON exits if first part of data file is not printable
func isJSON(head []byte) {
	for _, character := range head {
		if !(character > 31 && character < 126 || character == 9 || character == 10) {
			usageError("Data file invalid, read character:", string(character))
		}
	}
}

// checkModelPath checks the first part of data file, protects against /dev/random
func checkModelPath(filepath string) {
	f, err := os.Open(filepath)
	checkError("os.Open", err)
	reader := bufio.NewReader(f)
	head, err := reader.Peek(420)
	checkError("reader.Peek", err)
	isJSON(head)
}

// checkModel exits if model & neural network architecture don't match
func checkModel(model [][]jsonNeuron, nn neuralNetwork) {
	var architecture []int
	for layer := 0; layer < len(model); layer++ {
		architecture = append(architecture, len(model[layer]))
	}
	if len(model) != len(nn.architecture)-1 {
		fmt.Printf("%vERROR Neural network architecture don't match models: %v%v\n", RED, architecture, RESET)
		os.Exit(1)
	}
	for layer := 1; layer < len(nn.architecture); layer++ {
		if nn.architecture[layer] != len(model[layer-1]) {
			fmt.Printf("%vERROR Neural network architecture don't match models: %v%v\n", RED, architecture, RESET)
			os.Exit(1)
		}
		for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
			if len(nn.layers[layer].neurons[neuron].weights) != len(model[layer-1][neuron].Weights) {
				fmt.Printf("%vERROR Models architecture invalid, len(model[%v][%v].Weights): %v%v\n", RED, layer, neuron, len(model[layer-1][neuron].Weights), RESET)
				os.Exit(1)
			}
		}
	}
}

// Load bias & weights from model.json
func loadModel(nn neuralNetwork, filepath string) {
	fmt.Printf("Loading model from: %v...", filepath)

	checkModelPath(filepath)
	file, err := ioutil.ReadFile(filepath)
	checkError("ioutil.ReadFile", err)

	model := [][]jsonNeuron{}
	err = json.Unmarshal([]byte(file), &model)
	checkError("json.Unmarshal", err)
	checkModel(model, nn)

	for layer := 1; layer < len(nn.architecture); layer++ {
		for neuron := 0; neuron < nn.architecture[layer]; neuron++ {
			nn.layers[layer].neurons[neuron].bias = model[layer-1][neuron].Bias
			for weight := 0; weight < nn.architecture[layer-1]; weight++ {
				nn.layers[layer].neurons[neuron].weights[weight] = model[layer-1][neuron].Weights[weight]
			}
		}
	}
	fmt.Printf("\rModel loaded from: %v    \n\n", filepath)
}
