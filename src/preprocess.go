package multilayer

import (
	"fmt"
	"os"
	"io"
	"encoding/csv"
)

func readCsv(filePath string) /*[][]float64*/ {
    f, err := os.Open(filePath)
    if err != nil {
        fmt.Println("Unable to read input file " + filePath, err)
    }
    defer f.Close()

    csvReader := csv.NewReader(f)

	// // in := "id, diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst"

	// var data [][]float64
	for {
    	dataStr, err := csvReader.Read() // test for weakness e.g. /dev/null
		if err == io.EOF {
			break
		}
    	if err != nil {
    	    fmt.Println("Unable to parse file as CSV for " + filePath, err)
			os.Exit(1)
    	}
    	fmt.Println(dataStr) ////
		// break //////////////////////
	}
    // fmt.Println(dataStr[0][0]) ////
    // return data
}


func preprocess() int {
	// fmt.Printf("Oh hi!!!\n") ////////////
	readCsv("data.csv") ////
	// data := readCsv("data.csv")
    // fmt.Println(data[0][0]) ////

	return 42
}
