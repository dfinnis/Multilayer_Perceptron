package multilayer

import "math/rand"

// shuffle randomizes the order of the data samples
func shuffle(data [][]float32) {
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
}

// split splits data into training & test sets
func split(data [][]float32) (train_set [][]float32, test_set [][]float32) {
	var split float32 = 0.8
	var sample int
	for ; sample < int((float32(len(data)) * split)); sample++ {
		train_set = append(train_set, data[sample])
	}
	for ; sample < len(data); sample++ {
		test_set = append(test_set, data[sample])
	}
	return
}

// splitData shuffles data & creates training & test sets
func splitData(data [][]float32, flags flags) (train_set, test_set [][]float32) {
	shuffle(data)
	if flags.flagT && !flags.flagP && !flags.flagS && !flags.flagE {
		train_set = data
	} else if flags.flagP && !flags.flagT && !flags.flagS {
		test_set = data
	} else {
		train_set, test_set = split(data)
	}
	printSplit(len(train_set), len(test_set))
	return
}

// getX returns the data minus diagnosis
func getX(data [][]float32, i int) []float32 {
	var sample []float32
	for column := 1; column < len(data[0]); column++ {
		sample = append(sample, data[i][column])
	}
	return sample
}

// getY returns one hot diagnosis (Malignant / Benign)
func getY(data [][]float32, i int) []float32 {
	var oneHot []float32
	if data[i][0] == 0.0 {
		oneHot = []float32{0, 1}
	} else {
		oneHot = []float32{1, 0}
	}
	return oneHot
}

// splitXY splits sample data (x), & diagnosis (y)
func splitXY(data [][]float32) (x [][]float32, y [][]float32) {
	for i := 0; i < len(data); i++ {
		x = append(x, getX(data, i))
		y = append(y, getY(data, i))
	}
	return
}
