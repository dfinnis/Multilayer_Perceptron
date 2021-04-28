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
