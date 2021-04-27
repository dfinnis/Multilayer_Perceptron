package multilayer

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// parseArchitecture converts string to list of ints
func parseArchitecture(arg string) []int {
	var architecture []int
	list := strings.Fields(arg)
	for _, layer := range list {
		integer, err := strconv.Atoi(layer)
		if err != nil {
			usageError("Bad argument: ", arg)
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

// parseSeed converts string to int
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

// parseArg parses and returns arguments for flags -t -p -a -s
func parseArg() (flagT bool, flagP bool, filepath string, architecture []int, seed int64) {
	// Default settings
	filepath = "model.json"
	// architecture = []int{16, 16, 16, 2}
	architecture = []int{16, 16, 2}
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
			i++
			seed = parseSeed(i, args)
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
