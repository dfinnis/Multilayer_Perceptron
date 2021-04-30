package multilayer

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// defaultConfig initializes default values
func defaultConfig() (string, string, []int, int64, bool) {
	dataPath := "data.csv"
	modelPath := "model.json"
	architecture := []int{16, 16, 2}
	seed := time.Now().UnixNano()
	flagS := false
	return dataPath, modelPath, architecture, seed, flagS
}

// parseFilepath checks if filepath exists
func parseFilepath(filepath string) string {
	_, err := os.Stat(filepath)
	if err != nil {
		usageError("Invalid filepath: ", filepath)
	}
	return filepath
}

// parseArchitecture converts string to list of ints
func parseArchitecture(i int, args []string) []int {
	if i >= len(args) {
		usageError("No architecture provided after -a", "")
	}
	var architecture []int
	list := strings.Fields(args[i])
	for _, layer := range list {
		integer, err := strconv.Atoi(layer)
		if err != nil {
			usageError("Bad argument: ", args[i])
		}
		if integer > 100 {
			usageError("Max 100 neurons in a layer, given:", strconv.Itoa(integer))
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

// parseSeed converts arg string to int
func parseSeed(i int, args []string) int64 {
	if i >= len(args) {
		usageError("No seed provided after -s", "")
	}
	seed, err := strconv.Atoi(args[i])
	if err != nil {
		usageError("Bad seed: ", args[i])
	}
	return int64(seed)
}

// seedRandom initializes rand with time or -s SEED
func seedRandom(seed int64, flagS bool) {
	rand.Seed(seed)
	if !flagS {
		fmt.Printf("Random seed: %d\n\n", seed)
	}
}

// isFlag returns true if argument is flag (apart from -p)
func isFlag(arg string) bool {
	if arg == "-t" || arg == "--train" ||
		arg == "-a" || arg == "--architecture" ||
		arg == "-s" || arg == "--seed" ||
		arg == "-e" || arg == "--early" ||
		arg == "-mse" || arg == "--mean" ||
		arg == "-rmse" || arg == "--root" {
		return true
	}
	return false
}

// parseArg parses and returns arguments for flags -t -p -a -s
func parseArg() (flagT bool, dataPath string, flagP bool, modelPath string, architecture []int, flagE, flagS, flagMSE, flagRMSE bool, err error) {
	dataPath, modelPath, architecture, seed, flagS := defaultConfig()
	_, err = os.Stat(modelPath)

	args := os.Args[1:]
	if len(args) == 0 {
		seedRandom(seed, flagS)
		if err != nil {
			flagT = true
			flagP = true
		}
		return
	} else if len(args) > 8 {
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
				if isFlag(args[i+1]) {
					continue
				}
				i++
				modelPath = parseFilepath(args[i])
				_, err = os.Stat(modelPath)
			}
		} else if args[i] == "-a" || args[i] == "--architecture" {
			i++
			architecture = parseArchitecture(i, args)
		} else if args[i] == "-s" || args[i] == "--seed" {
			i++
			seed = parseSeed(i, args)
			flagS = true
		} else if args[i] == "-e" || args[i] == "--early" {
			flagE = true
		} else if args[i] == "-mse" || args[i] == "--mean" {
			flagMSE = true
		} else if args[i] == "-rmse" || args[i] == "--root" {
			flagRMSE = true
		} else {
			dataPath = parseFilepath(args[i])
		}
	}

	if flagT && flagP && modelPath != "model.json" {
		errorExit("invalid option combination: -t saves model.json but -p loads different model")
	}
	if flagT && !flagP && flagE && !flagS {
		flagE = false
	}
	seedRandom(seed, flagS)
	return
}
