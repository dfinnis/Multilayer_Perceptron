package multilayer

import (
	"fmt"
	"os"
)

func printUsage() {
	fmt.Printf("\nUsage:\tgo build; ./Multilayer_Perceptron [-t] [-p [FILEPATH]] [-a ARCHITECTURE] [-s SEED]\n\n")
	fmt.Printf("    [-t] (--train)        Only train, don't predict. Overwrites existing model\n")
	fmt.Printf("    [-p] (--predict)      Only predict, don't train. Optional [FILEPATH] load model from filepath\n")
	fmt.Printf("    [-a] (--architecture) Provide ARCHITECTURE as string e.g. \"16 16 16 2\"\n")
	fmt.Printf("    [-s] (--seed)         Provide SEED for randomization\n")
	fmt.Printf("    [-h] (--help)         Show usage\n\n")
	os.Exit(1)
}

func usageError(msg, err string) {
	fmt.Printf("ERROR %v %v\n", msg, err)
	printUsage()
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
