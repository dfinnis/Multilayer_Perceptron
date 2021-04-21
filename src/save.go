package multilayer

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

// Models can be saved and loaded, json file <=> 2D array [][]jsonNeuron
type jsonNeuron struct {
	Bias    float64
	Weights []float64
}

func errorExit(message string) {
	fmt.Printf("ERROR %v\n", message)
	os.Exit(1)
}

func checkError(message string, err error) {
	if err != nil {
		fmt.Printf("ERROR %v %v\n", message, err)
		os.Exit(1)
	}
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
	ioutil.WriteFile("model.json", jsonString, 0644)
}

// checkModel exits if model & neural network architecture don't match
func checkModel(model [][]jsonNeuron, nn neuralNetwork) {
	if len(model) != len(nn.architecture)-1 {
		errorExit("Model & neural network architecture different")
	}
	for layer := 1; layer < len(nn.architecture); layer++ {
		if nn.architecture[layer] != len(model[layer-1]) {
			errorExit("Model & neural network architecture different")
		}
	}
}

// Load bias & weights from model.json
func loadModel(nn neuralNetwork) {
	file, err := ioutil.ReadFile("model.json")
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
}
