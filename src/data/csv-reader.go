package data

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

func readCsv(filename string, startIndex, endIndex, limit int) [][]float64 {
    file := readFile(filename)
    defer file.Close()
    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
    	log.Fatal("Could not read CSV file", filename)
	}

    if startIndex == -1 {
    	startIndex = 0
	}

    if limit == -1 {
    	limit = len(records)
	}

	data := make([][]float64, limit)
    for i, row := range records {
    	if i >= limit {
    		break
		}

		if endIndex == -1 {
			endIndex = len(row) - 1
		}

    	data[i] = make([]float64, endIndex - startIndex + 1)
    	associationIndex := 0
    	for j, value := range row {
    		if j < startIndex || j > endIndex {
    			continue
			}

            convertedValue, err := strconv.ParseFloat(value, 64)
            if err != nil {
            	log.Fatal("Could not parse one of the values in the CSV file", filename)
			}
            data[i][associationIndex] = convertedValue
            associationIndex++
		}
	}

	return data
}

func readFile(filename string) *os.File {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal("Could not open file", filename)
	}
	return file
}
