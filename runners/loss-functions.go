package runners

import (
	"math"
)

func squareError(outputs, expected [][]float64) []float64 {
	errors := make([]float64, len(outputs))
	for i, output := range outputs {
		for j, y := range output {
			errors[i] += math.Pow(y - expected[i][j], 2)
		}
		errors[i] /= 2
	}
	return errors
}
