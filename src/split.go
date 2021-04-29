package multilayer

import (
	"math/rand"
)

// shuffle randomizes the order of the data samples
func shuffle(data [][]float64) {
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
}

// split splits data into training and test sets
func split(data [][]float64) (train_set [][]float64, test_set [][]float64) {
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

// splitData shuffles data and creates training and test sets
func splitData(data [][]float64, flagT, flagP, flagS bool, err error) (train_set, test_set [][]float64) {
	shuffle(data)
	if (!flagT && !flagP && flagS) || (flagT && flagP) || flagS || err != nil {
		train_set, test_set = split(data)
	} else if flagT {
		train_set = data
	} else { // flagP
		test_set = data
	}
	printSplit(len(train_set), len(test_set))
	return
}

// split_x_y splits sample data (x), & diagnosis (y)
func split_x_y(data [][]float64) (x [][]float64, y [][]float64) {
	for i := 0; i < len(data); i++ {
		// x = data
		var sample []float64
		for column := 1; column < len(data[0]); column++ {
			sample = append(sample, data[i][column])
		}
		x = append(x, sample)
		// y = one hot diagnosis (Malignant / Benign)
		var oneHot []float64
		if data[i][0] == 0.0 {
			oneHot = []float64{0, 1}
		} else {
			oneHot = []float64{1, 0}
		}
		y = append(y, oneHot)
	}
	return
}
