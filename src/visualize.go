package multilayer

import (
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
)

// lossToStr converts 2 lists of floats to a 2d string array
func lossToStr(trainLoss, testLoss []float64) [][]string {
	var lossStrArray [][]string
	for i := 0; i < len(trainLoss); i++ {
		var lossStr []string
		lossStr = append(lossStr, fmt.Sprintf("%g", trainLoss[i]))
		lossStr = append(lossStr, fmt.Sprintf("%g", testLoss[i]))
		lossStrArray = append(lossStrArray, lossStr)
	}
	return lossStrArray
}

// writeCSV writes trainLoss & testLoss to loss.csv
func writeCSV(trainLoss, testLoss []float64) {
	loss := lossToStr(trainLoss, testLoss)
	file, err := os.Create("loss.csv")
	checkError("Cannot create file", err)
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, value := range loss {
		err := writer.Write(value)
		checkError("Cannot write to file", err)
	}
}

// visualize calls visualize.py, giving loss as argument
func visualize(trainLoss, testLoss []float64) {
	writeCSV(trainLoss, testLoss)
	cmd := exec.Command("python3", "src/visualize.py")
	out, err := cmd.CombinedOutput()
	checkError("visualize.py", err)
	if len(out) != 0 {
		errorExit("visualize.py")
	}
	err = os.Remove("loss.csv")
	checkError("Failed to delete loss.csv", err)
}
