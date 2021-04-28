package multilayer

import (
	"fmt"
	"os"
)

func printUsage() {
	fmt.Printf("\nUsage:\tgo build; ./Multilayer_Perceptron [-t] [-p [FILEPATH]] [-a ARCHITECTURE] [-s SEED]\n\n")
	fmt.Printf("    [-t] (--train)        Only train, don't predict. Overwrites existing model\n")
	fmt.Printf("    [-p] (--predict)      Only predict, don't train. Optional [FILEPATH] load model from filepath\n")
	fmt.Printf("    [-a] (--architecture) Provide ARCHITECTURE as string e.g. -a \"16 16 2\"\n")
	fmt.Printf("    [-s] (--seed)         Provide SEED integer for randomization e.g. -s 42\n")
	fmt.Printf("    [-h] (--help)         Show usage\n\n")
	os.Exit(1)
}

func usageError(msg, err string) {
	fmt.Printf("%vERROR %v %v%v\n", RED, msg, err, RESET)
	printUsage()
}

func errorExit(message string) {
	fmt.Printf("%vERROR %v%v\n", RED, message, RESET)
	os.Exit(1)
}

func checkError(message string, err error) {
	if err != nil {
		fmt.Printf("%vERROR %v %v%v\n", RED, message, err, RESET)
		os.Exit(1)
	}
}
