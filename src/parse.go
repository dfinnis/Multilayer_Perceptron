package multilayer

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// flags contains all flags & arguments
type flags struct {
	dataPath     string
	dataPathSet  bool
	flagT        bool
	flagP        bool
	modelPath    string
	architecture []int
	seed         int64
	epochs       int
	learningRate float32
	flagE        bool
	flagS        bool
	flagMSE      bool
	flagRMSE     bool
	flagQ        bool
}

// defaultConfig initializes default values
func defaultConfig() flags {
	flags := flags{}
	flags.dataPath = "data.csv"
	flags.modelPath = "model.json"
	flags.architecture = []int{16, 16, 2}
	flags.seed = time.Now().UnixNano()
	flags.learningRate = 0.01
	flags.epochs = 15000
	return flags
}

// isFlag returns true if argument is flag (apart from -p)
func isFlag(arg string) bool {
	if arg == "-t" || arg == "--train" ||
		arg == "-a" || arg == "--architecture" ||
		arg == "-s" || arg == "--seed" ||
		arg == "-e" || arg == "--early" ||
		arg == "-mse" || arg == "--mean" ||
		arg == "-rmse" || arg == "--root" ||
		arg == "-l" || arg == "--learning" ||
		arg == "-ep" || arg == "--epochs" ||
		arg == "-q" || arg == "--quiet" {
		return true
	}
	return false
}

// parseFilepath checks if filepath exists
func parseFilepath(filepath string) string {
	_, err := os.Stat(filepath)
	if err != nil {
		usageError("Invalid filepath: ", filepath)
	}
	return filepath
}

// parseDataPath sets path to data, default: data.csv. Catches all bad arguments
func parseDataPath(filepath string, flags flags) flags {
	if flags.dataPathSet {
		usageError("Invalid argument: ", filepath)
	}
	flags.dataPath = parseFilepath(filepath)
	flags.dataPathSet = true
	return flags
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
		if integer < 2 {
			usageError("Min 2 neurons in a layer, given:", strconv.Itoa(integer))
		}
		architecture = append(architecture, integer)
	}
	if len(architecture) < 2 {
		usageError("Architecture minimum 1 hidden layer", "")
	}
	if len(architecture) > 3 {
		usageError("Architecture maximum 2 hidden layers", "")
	}
	if architecture[len(architecture)-1] != 2 {
		usageError("Architecture invalid, output layer must be 2 neurons", "")
	}
	return architecture
}

// parseSeed converts arg string to int
func parseSeed(i int, args []string, flags flags) flags {
	if i >= len(args) {
		usageError("No seed provided after -s", "")
	}
	seed, err := strconv.Atoi(args[i])
	if err != nil {
		usageError("Bad seed: ", args[i])
	}
	flags.seed = int64(seed)
	flags.flagS = true
	return flags
}

// parseEpochs parses string to int, must be between 0 & 100000
func parseEpochs(i int, args []string) int {
	if i >= len(args) {
		usageError("No epochs number provided after -ep", "")
	}
	epochs, err := strconv.Atoi(args[i])
	if err != nil {
		usageError("Bad epochs: ", args[i])
	}
	if epochs <= 0 || epochs >= 100000 {
		usageError("Learning rate must be between 0 & 100000, given: ", args[i])
	}
	return epochs
}

// parseLearningRate parses string to float, must be between 0 & 1
func parseLearningRate(i int, args []string) float32 {
	if i >= len(args) {
		usageError("No learning rate provided after -l", "")
	}
	learningRate, err := strconv.ParseFloat(args[i], 32)
	if err != nil {
		usageError("Bad learning rate: ", args[i])
	}
	if learningRate <= 0 || learningRate >= 1 {
		usageError("Learning rate must be between 0 & 1, given: ", args[i])
	}
	return float32(learningRate)
}

// parseArg parses & returns arguments for flags
func parseArg() flags {
	flags := defaultConfig()

	args := os.Args[1:]
	if len(args) == 0 {
		return flags
	} else if len(args) > 13 {
		usageError("Too many arguments: ", strconv.Itoa(len(args)))
	}

	for i := 0; i < len(args); i++ {
		if args[i] == "-h" || args[i] == "--help" {
			printUsage()
		} else if args[i] == "-t" || args[i] == "--train" {
			flags.flagT = true
		} else if args[i] == "-p" || args[i] == "--predict" {
			flags.flagP = true
			if i < len(args)-1 {
				if isFlag(args[i+1]) {
					continue
				}
				i++
				flags.modelPath = parseFilepath(args[i])
			}
		} else if args[i] == "-a" || args[i] == "--architecture" {
			i++
			flags.architecture = parseArchitecture(i, args)
		} else if args[i] == "-s" || args[i] == "--seed" {
			i++
			flags = parseSeed(i, args, flags)
		} else if args[i] == "-e" || args[i] == "--early" {
			flags.flagE = true
		} else if args[i] == "-ep" || args[i] == "--epochs" {
			i++
			flags.epochs = parseEpochs(i, args)
		} else if args[i] == "-l" || args[i] == "--learning" {
			i++
			flags.learningRate = parseLearningRate(i, args)
		} else if args[i] == "-mse" || args[i] == "--mean" {
			flags.flagMSE = true
		} else if args[i] == "-rmse" || args[i] == "--root" {
			flags.flagRMSE = true
		} else if args[i] == "-q" || args[i] == "--quiet" {
			flags.flagQ = true
		} else {
			flags = parseDataPath(args[i], flags)
		}
	}

	if flags.flagT && flags.flagP && flags.modelPath != "model.json" {
		errorExit("invalid option combination: -t saves model.json but -p loads different model")
	}
	return flags
}
