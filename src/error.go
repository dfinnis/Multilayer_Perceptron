package multilayer

import (
	"fmt"
	"os"
)

// printUsage prints usage & quits
func printUsage() {
	fmt.Printf("\nUsage:\tgo build; ./Multilayer_Perceptron [DATA.CSV] [-t] [-p [FILEPATH]] [-s SEED] [-e] [-ep EPOCHS] [-l LEARNING] [-a ARCHITECTURE] [-mse] [-rmse] [-q] [-h]\n\n")
	fmt.Printf("    [-t]    (--train)        Without -p or -e or -s, -t uses the entire dataset to train\n")
	fmt.Printf("    [-p]    (--predict)      Only predict, don't train. Optional [FILEPATH] load model from filepath\n")
	fmt.Printf("    [-s]    (--seed)         Provide SEED integer for randomization e.g. -s 42\n")
	fmt.Printf("    [-e]    (--early)        Early stopping. Stop training when test set loss starts increasing\n")
	fmt.Printf("    [-ep]   (--epochs)       Provide EPOCHS. Must be integer between 0 & 100000, default 15000\n")
	fmt.Printf("    [-l]    (--learning)     Provide LEARNING rate. Must be float between 0 & 1, default 0.01\n")
	fmt.Printf("    [-a]    (--architecture) Provide ARCHITECTURE as string e.g. -a \"16 16 2\"\n")
	fmt.Printf("    [-mse]  (--mean)         Loss metric mean squared error\n")
	fmt.Printf("    [-rmse] (--root)         Loss metric root mean squared error\n")
	fmt.Printf("    [-q]    (--quiet)        Don't print architecture or seed or additional metrics while training\n")
	fmt.Printf("    [-h]    (--help)         Show usage\n\n")
	os.Exit(1)
}

// usageError prints error message & usage, then quits
func usageError(message, err string) {
	fmt.Printf("%vERROR %v %v%v\n", RED, message, err, RESET)
	printUsage()
}

// errorExit prints error message & quits
func errorExit(message string) {
	fmt.Printf("%vERROR %v%v\n", RED, message, RESET)
	os.Exit(1)
}

// checkError prints error message & quits if error
func checkError(message string, err error) {
	if err != nil {
		fmt.Printf("%vERROR %v %v%v\n", RED, message, err, RESET)
		os.Exit(1)
	}
}
