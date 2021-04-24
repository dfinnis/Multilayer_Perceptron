package multilayer

import (
	"fmt"
	"os"
)

func printUsage() {
	fmt.Printf("\nUsage:\tgo build; ./Multilayer_Perceptron [-t] [-p [filepath]]\n\n")
	fmt.Printf("    [-t] (--train)   Only train, don't predict. Overwrites existing model\n")
	fmt.Printf("    [-p] (--predict) Only predict, don't train. Optional [filepath] load model from filepath\n")
	fmt.Printf("    [-h] (--help)    Show usage\n\n")
	os.Exit(1)
}

// parseArg parses arguments, returns mix and flags
func parseArg() (flagT bool, flagP bool, filepath string) {
	filepath = "model.json"
	args := os.Args[1:]
	if len(args) == 0 {
		return false, true, filepath
	} else if len(args) > 3 {
		fmt.Printf("ERROR Too many arguments\n")
		printUsage()
	}
	for i := 0; i < len(args); i++ {
		if args[i] == "-h" || args[i] == "--help" {
			printUsage()
		} else if args[i] == "-t" || args[i] == "--train" {
			flagT = true
		} else if args[i] == "-p" || args[i] == "--predict" {
			flagP = true
			if i < len(args)-1 {
				i++
				filepath = args[i]
				_, err := os.Stat(filepath)
				if err != nil {
					fmt.Printf("ERROR Invalid model filepath\n")
					printUsage()
				}
			}
		} else {
			fmt.Printf("Error bad argument: %v\n", args[i])
			printUsage()
		}
	}
	if flagT && flagP && filepath != "model.json" {
		errorExit("invalid option combination: -t saves model.json but -p loads different model")
	}
	return
}
