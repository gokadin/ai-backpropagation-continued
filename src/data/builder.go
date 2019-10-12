package data

import (
	"log"
	"sort"
)

type builder struct {
	dataset *Dataset
}

func newBuilder(dataset *Dataset) *builder {
	return &builder{
		dataset: dataset,
	}
}

func (b *builder) readCsv(filename string, startIndex, endIndex, limit int) *builder {
	b.dataset.data = readCsv(filename, startIndex, endIndex, limit)
	return b
}

func (b *builder) readRandom(associations, size int) *builder {
    b.dataset.data = readRandom(associations, size)
    return b
}

func (b *builder) Normalize(min, max float64, columns ...int) *builder {
	if len(columns) == 0 {
		b.normalizeAll(min, max)
	} else {
		b.normalizeForColumns(min, max, columns)
	}
	return b
}

func (b *builder) normalizeForColumns(min, max float64, columns []int) {
	for _, row := range b.dataset.data {
		for _, columnIndex := range columns {
			row[columnIndex] = (row[columnIndex] - min) / (max - min)
		}
	}
}

func (b *builder) normalizeAll(min, max float64) {
	for _, row := range b.dataset.data {
        for j := 0; j < len(row); j++ {
			row[j] = (row[j] - min) / (max - min)
		}
	}
}

func (b *builder) OneHotEncode() *builder {
	if b.dataset.Size() == 0 || len(b.dataset.data[0]) != 1 {
        log.Fatal("cannot one hot encode dataset with more than one value per output")
	}

	possibleValues := make(map[float64]int)
    for _, output := range b.dataset.data {
		possibleValues[output[0]] = 1
	}
    keys := make([]float64, 0)
    for key := range possibleValues {
    	keys = append(keys, key)
	}
    sort.Float64s(keys)

	for i, output := range b.dataset.data {
		encoded := make([]float64, len(possibleValues))
		counter := 0
		for _, value := range keys {
            if output[0] == value {
                encoded[counter] = 1.0
			} else {
				encoded[counter] = 0.0
			}
            counter++
		}
		b.dataset.data[i] = encoded
	}

    return b
}