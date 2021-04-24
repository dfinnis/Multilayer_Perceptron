package multilayer

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

func printUsage() {
	fmt.Printf("\nUsage:\tgo build; ./Multilayer_Perceptron [-t] [-p [filepath]]\n\n")
	fmt.Printf("    [-t] (--train)   Only train, don't predict. Overwrites existing model\n")
	fmt.Printf("    [-p] (--predict) Only predict, don't train. Optional [filepath] load model from filepath\n")
	fmt.Printf("    [-h] (--help)    Show usage\n\n")
	os.Exit(1)
}

func parseArchitecture(arg string) []int {
	var architecture []int
	list := strings.Fields(arg)
	for _, layer := range list {
		integer, err := strconv.Atoi(layer)
		if err != nil {
			fmt.Printf("ERROR Bad argument: %v\n", arg)
			printUsage()
		}
		architecture = append(architecture, integer)
	}
	if len(architecture) < 2 {
		fmt.Printf("ERROR Architecture minimum 2 layers\n")
		printUsage()
	}
	return architecture
}

// parseArg parses arguments, returns mix and flags
func parseArg() (flagT bool, flagP bool, filepath string, architecture []int) {
	filepath = "model.json"
	architecture = []int{16, 16, 16, 2}

	args := os.Args[1:]
	if len(args) == 0 {
		return false, true, filepath, architecture
	} else if len(args) > 4 {
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
				if args[i] == "-t" || args[i] == "--train" {
					flagT = true
					continue
				}
				filepath = args[i]
				_, err := os.Stat(filepath)
				if err != nil {
					fmt.Printf("ERROR Invalid model filepath\n")
					printUsage()
				}
			}
		} else if args[i] == "-a" || args[i] == "--architecture" {
			if i < len(args)-1 {
				i++
				architecture = parseArchitecture(args[i])
			} else {
				fmt.Printf("ERROR No architecture provided after -a\n")
				printUsage()
			}
		} else {
			fmt.Printf("ERROR Bad argument: %v\n", args[i])
			printUsage()
		}
	}
	if flagT && flagP && filepath != "model.json" {
		errorExit("invalid option combination: -t saves model.json but -p loads different model")
	}
	if !flagT && !flagP {
		flagP = true
	}
	return
}
