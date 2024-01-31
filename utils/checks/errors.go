package checks

import (
	"github.com/phuslu/log"
	"runtime/debug"
	"strings"
)

// Checks has its own package, to prevent dependency cycles

func Check(err error) {
	if err != nil {
		stack := strings.Join(strings.Split(string(debug.Stack()), "\n")[5:], "\n")
		log.Fatal().Stack().Err(err).Msg(stack)
	}
}

func CheckWithMessage(err error, message string) {
	if err != nil {
		stack := strings.Join(strings.Split(string(debug.Stack()), "\n")[5:], "\n")
		log.Fatal().Stack().Err(err).Str("stack", stack).Msg(message)
	}
}
