package multilayer

import (
	"fmt"
	"os/exec"
)

// lossToStr converts 2 lists of floats to a string
func lossToStr(trainLoss, testLoss []float64) string {
	var lossStr string
	for _, loss := range trainLoss {
		lossStr = lossStr + fmt.Sprintf("%g", loss) + " "
	}
	for i, loss := range testLoss {
		lossStr = lossStr + fmt.Sprintf("%g", loss)
		if i != len(testLoss)-1 {
			lossStr = lossStr + " "
		}
	}
	return lossStr
}

// visualize calls visualize.py, giving loss as argument
func visualize(trainLoss, testLoss []float64) {
	lossStr := lossToStr(trainLoss, testLoss)
	cmd := exec.Command("python3", "src/visualize.py", lossStr)
	out, err := cmd.CombinedOutput()
	checkError("visualize.py", err)
	if len(out) != 0 {
		errorExit("visualize.py")
	}
}
