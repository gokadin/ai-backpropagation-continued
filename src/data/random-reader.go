package data

import "math/rand"

func readRandom(associations, size int) [][]float64 {
	data := make([][]float64, associations)
    for i := 0; i < associations; i++ {
		association := make([]float64, size)
        for j := 0; j < size; j++ {
			association[j] = rand.Float64()
		}
        data[i] = association
	}
    return data
}

