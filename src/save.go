package multilayer

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

// Save/load model, 2D array [][]jsonNeuron <=> json file
type jsonNeuron struct {
	Bias    float64
	Weights []float64
}

// Save bias & weights to model.json
func saveModel(nn neuralNetwork) {
	fmt.Printf("Saving model...\r")
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
	ioutil.WriteFile("model.json", jsonString, 0644)
	fmt.Printf("Model saved as model.json\n\n")
}

// checkModel exits if model & neural network architecture don't match
func checkModel(model [][]jsonNeuron, nn neuralNetwork) {
	var architecture []int
	for layer := 0; layer < len(model); layer++ {
		architecture = append(architecture, len(model[layer]))
	}
	if len(model) != len(nn.architecture)-1 {
		fmt.Printf("ERROR Neural network architecture don't match models: %v\n", architecture)
		os.Exit(1)
	}
	for layer := 1; layer < len(nn.architecture); layer++ {
		if nn.architecture[layer] != len(model[layer-1]) {
			fmt.Printf("ERROR Neural network architecture don't match models: %v\n", architecture)
			os.Exit(1)
		}
	}
}

// Load bias & weights from model.json
func loadModel(nn neuralNetwork, filepath string) {
	fmt.Printf("Loading model from %v...", filepath)
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
	fmt.Printf("\rModel loaded from %v    \n\n", filepath)
}
