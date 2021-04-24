package multilayer

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
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

func parseArchitecture(arg string) []int {
	var architecture []int
	list := strings.Fields(arg)
	for _, layer := range list {
		integer, err := strconv.Atoi(layer)
		if err != nil {
			usageError("Bad argument: ", arg)
		}
		architecture = append(architecture, integer)
	}
	if len(architecture) < 2 {
		usageError("Architecture minimum 2 layers", "")
	}
	if architecture[len(architecture)-1] != 2 {
		usageError("Architecture invalid, output layer must be 2 neurons", "")
	}
	return architecture
}

func parseSeed(i int, args []string) int64 {
	var seed int64
	i++
	if i < len(args) {
		seedInt, err := strconv.Atoi(args[i])
		if err != nil {
			usageError("Bad seed: ", args[i])
		}
		seed = int64(seedInt)
	} else {
		usageError("No seed provided after -s", "")
	}
	return seed
}

// parseArg parses arguments, returns mix and flags
func parseArg() (flagT bool, flagP bool, filepath string, architecture []int, seed int64) {
	filepath = "model.json"
	architecture = []int{16, 16, 16, 2}
	seed = time.Now().UnixNano()

	args := os.Args[1:]
	if len(args) == 0 {
		return false, true, filepath, architecture, seed
	} else if len(args) > 6 {
		usageError("Too many arguments: ", strconv.Itoa(len(args)))
	}
	for i := 0; i < len(args); i++ {
		if args[i] == "-h" || args[i] == "--help" {
			printUsage()
		} else if args[i] == "-t" || args[i] == "--train" {
			flagT = true
		} else if args[i] == "-p" || args[i] == "--predict" {
			flagP = true
			if i < len(args)-1 {
				if args[i+1] == "-t" || args[i+1] == "--train" || args[i+1] == "-a" || args[i+1] == "--architecture" || args[i+1] == "-s" || args[i+1] == "--seed" {
					continue
				}
				i++
				filepath = args[i]
				_, err := os.Stat(filepath)
				if err != nil {
					usageError("Invalid model filepath: ", filepath)
				}
			}
		} else if args[i] == "-a" || args[i] == "--architecture" {
			if i < len(args)-1 {
				i++
				architecture = parseArchitecture(args[i])
			} else {
				usageError("No architecture provided after -a", "")
			}
		} else if args[i] == "-s" || args[i] == "--seed" {
			seed = parseSeed(i, args)
			i++
		} else {
			usageError("Bad argument: ", args[i])
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
