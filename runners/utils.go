package runners

import "math/rand"

func shuffleDataset(a, b [][]float64) {
	rand.Shuffle(len(a), func(i, j int) {
		a[i], a[j] = a[j], a[i]
		b[i], b[j] = b[j], b[i]
	})
}

func mean(arr []float64) float64 {
	mean := 0.0
    for _, value := range arr {
        mean += value
	}
    return mean / float64(len(arr))
}

func partitionData(data [][]float64, batchCounter, batchSize int) [][]float64 {
	fromIndex := batchCounter * batchSize
	toIndex := (batchCounter + 1) * batchSize
	if toIndex > len(data) {
		toIndex = len(data)
	}

	return data[fromIndex:toIndex]
}
