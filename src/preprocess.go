package multilayer

import (
	"encoding/csv"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/stat"
)

// readCsv reads data.csv into a 2d array of floats
func readCsv(filePath string) [][]float64 {
	f, err := os.Open(filePath)
	checkError("Unable to read input data file", err)
	defer f.Close()
	csvReader := csv.NewReader(f)
	// // in := "id, diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst"

	var data [][]float64
	for {
		var drop bool
		dataStr, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		checkError("Unable to parse data file as CSV", err)
		var sample []float64
		for column, dataPoint := range dataStr {
			if column == 1 {
				if dataPoint == "M" {
					sample = append(sample, 1) // M = 1
				} else if dataPoint == "B" {
					sample = append(sample, 0) // B = 0
				} else {
					errorExit("Invalid data file format")
				}
			}
			if column > 1 {
				if dataPoint == "0" { // drop samples with empty data points
					drop = true
					break
				}
				float, err := strconv.ParseFloat(dataPoint, 64)
				checkError("Unable to parse data file as float", err)
				sample = append(sample, float)
			}
		}
		if !drop {
			data = append(data, sample)
		}
	}
	return data
}

// standardize centers data around mean, & applys a standard deviation
func standardize(data [][]float64) {
	for col := 1; col < len(data[0]); col++ {
		var column []float64
		for _, sample := range data {
			column = append(column, sample[col])
		}

		mean := stat.Mean(column, nil)
		variance := stat.Variance(column, nil)
		stddev := math.Sqrt(variance)

		for sample, _ := range data {
			data[sample][col] = (data[sample][col] - mean) / stddev
		}
	}
}

// preprocess reads data.csv & standardizes data
func preprocess() [][]float64 {
	data := readCsv("data.csv")
	standardize(data)
	return data
}

// shuffle randomizes the order of the data samples
func shuffle(data [][]float64) {
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
}

// split shuffles and splits data into the training and test set
func split(data [][]float64) (train_set [][]float64, test_set [][]float64) {
	shuffle(data)
	split := 0.8
	var sample int
	for ; sample < int((float64(len(data)) * split)); sample++ {
		train_set = append(train_set, data[sample])
	}
	for ; sample < len(data); sample++ {
		test_set = append(test_set, data[sample])
	}
	return
}

// split_x_y splits sample data (x), & diagnosis (y)
func split_x_y(train_set [][]float64) (x [][]float64, y [][]float64) {
	for i := 0; i < len(train_set); i++ {
		// x = data
		var sample []float64
		for column := 1; column < len(train_set[0]); column++ {
			sample = append(sample, train_set[i][column])
		}
		x = append(x, sample)
		// y = one hot diagnosis (Malignant / Benign)
		var oneHot []float64
		if train_set[i][0] == 0.0 {
			oneHot = []float64{0, 1}
		} else {
			oneHot = []float64{1, 0}
		}
		y = append(y, oneHot)
	}
	return
}
