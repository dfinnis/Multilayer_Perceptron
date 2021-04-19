package multilayer

func transpose(x [][]float64) [][]float64 {
	out := make([][]float64, len(x[0]))
	for i := 0; i < len(x); i += 1 {
		for j := 0; j < len(x[0]); j += 1 {
			out[j] = append(out[j], x[i][j])
		}
	}
	return out
}

func multiply(x, y [][]float64) [][]float64 {
	// fmt.Println("oh hi multiply!") /////////////////
	// fmt.Printf("len(x[0]): %v\n", len(x[0])) ////////////
	// fmt.Printf("len(y): %v\n", len(y)) ////////////
	if len(x[0]) != len(y) {
		return nil
	}
	out := make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = make([]float64, len(y[0]))
		for j := 0; j < len(y[0]); j++ {
			for k := 0; k < len(y); k++ {
				out[i][j] += x[i][k] * y[k][j]
			}
		}
	}
	return out
}

func multiply2(x, y [][]float64) [][]float64 {
	// fmt.Println("oh hi multiply2!") /////////////////
	// fmt.Printf("len(x): %v\n", len(x)) ////////////
	// fmt.Printf("len(x[0]): %v\n", len(x[0])) ////////////
	// fmt.Printf("len(y): %v\n", len(y)) ////////////
	// fmt.Printf("len(y[0]): %v\n", len(y[0])) ////////////
	// fmt.Printf("x[0][0]: %v\n", x[0][0]) ////////////
	// fmt.Printf("x[0][1]: %v\n", x[0][1]) ////////////
	// fmt.Printf("y[0][0]: %v\n", y[0][0]) ////////////
	// fmt.Printf("y[0][1]: %v\n", y[0][1]) ////////////
	out := make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		// fmt.Printf("i: %v\n", i) ////////////
		out[i] = make([]float64, len(y[0]))
		for j := 0; j < len(y[0]); j++ {
			// fmt.Printf("j: %v\n", j) ////////////
			for k := 0; k < len(y); k++ {
				// fmt.Printf("k: %v\n", k) ////////////
				out[i][j] += x[i][k] * y[k][j]
			}
		}
	}
	return out
}

// func matrixMultiply(a [][]float64, b [][]float64) (result [][]float64) {
// 	// fmt.Printf("\nlen(a): %v\n", len(a)) ////////////
// 	// fmt.Printf("len(a[0]): %v\n", len(a[0])) ////////////
// 	// fmt.Printf("len(b): %v\n", len(b)) ////////////
// 	// fmt.Printf("len(b[0]): %v\n", len(b[0])) ////////////
// 	if len(a) == 0 || len(b) == 0 { // protect against empty input
// 		return
// 	}
// 	for i := 0; i < len(a[0]); i++ {
// 		// fmt.Printf("i: %v\n", i) ///////////
// 		var row []float64
// 		for j := 0; j < len(b[0]); j++ {
// 			var sum float64
// 			// fmt.Printf("j: %v\n", j) ///////////
// 			for sample := 0; sample < len(a); sample++ {
// 				// fmt.Printf("sample: %v\n", sample) ////////////
// 				// fmt.Printf("a[%v][%v]: %v\n", sample, i, a[sample][i]) ////////////
// 				// fmt.Printf("b[%v][%v]: %v\n", sample, j, b[sample][j]) ////////////
// 				sum += a[sample][i] * b[sample][j]
// 			}
// 			row = append(row, sum)
// 		}
// 		result = append(result, row)
// 	}
// 	return
// }

// func matrixMultiplyTest() {
	// var a [][]float64
	// var b [][]float64
	// var tmp []float64
	// tmp = append(tmp, 2)
	// tmp = append(tmp, 0)
	// tmp = append(tmp, 1)
	// a = append(a, tmp)
	// tmp = nil
	// tmp = append(tmp, 1)
	// tmp = append(tmp, 1)
	// tmp = append(tmp, 1)
	// a = append(a, tmp)
	// tmp = nil
	// tmp = append(tmp, 1)
	// tmp = append(tmp, 1)
	// tmp = append(tmp, 1)
	// a = append(a, tmp)

	// tmp = nil
	// tmp = append(tmp, 2)
	// tmp = append(tmp, 1)
	// b = append(b, tmp)
	// tmp = nil
	// tmp = append(tmp, 1)
	// tmp = append(tmp, 1)
	// b = append(b, tmp)
	// b = append(b, tmp)
	// fmt.Printf("a: %v\n", a)
	// fmt.Printf("b: %v\n", b)
	// // Truth: [[6 4] [2 2] [4 3]]
	// fmt.Printf("matrixMultiply test: %v\n", matrixMultiply(a, b))
// }
