package multilayer

import (
	"fmt"
	"os"
	"io"
	"encoding/csv"
	"strconv"
	"gonum.org/v1/gonum/stat"
	"math"
)

func readCsv(filePath string) [][]float64 {
    f, err := os.Open(filePath)
    if err != nil {
        fmt.Println("ERROR Unable to read input file: " + filePath, err)
		os.Exit(1)
    }
    defer f.Close()
    csvReader := csv.NewReader(f)

	// // in := "id, diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst"

	var data [][]float64
	for {
		var drop bool
    	dataStr, err := csvReader.Read() // test for weakness e.g. /dev/null
		if err == io.EOF {
			break
		}
    	if err != nil {
    	    fmt.Println("ERROR Unable to parse file as CSV: " + filePath, err)
			os.Exit(1)
    	}
		var sample []float64
		for column, dataPoint := range dataStr {
			if column == 1 {
				if dataPoint == "M" {
					sample = append(sample, 1) // M = 1
				} else if dataPoint == "B" {
					sample = append(sample, 0) // B = 0
				} else {
					fmt.Println("ERROR invalid file format: " + filePath)
					os.Exit(1)
				}
			}			
			if column > 1 {
				if dataPoint == "0" { // drop samples with empty data points
					drop = true
					break
				}
				float, err := strconv.ParseFloat(dataPoint, 64)
				if err != nil {
					fmt.Println("ERROR Unable to parse file as float: " + filePath, err)
					os.Exit(1)
				}
				sample = append(sample, float)
			}
		}
		if !drop {
			data = append(data, sample)
		}
	}
    return data
}

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

func preprocess() [][]float64 {
	data := readCsv("data.csv") ////
	standardize(data)
	return data
}
