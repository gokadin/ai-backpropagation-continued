package data

import (
	"github.com/gokadin/ai-backpropagation-continued"
	"testing"
)

func TestDataset_GenerateRandomData(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

    ds.FromRandom(10, 1)

	if ds.Size() != 10 {
		t.Fatal("data size should be 10, got", ds.Size())
	}
}

func TestDataset_GenerateRandomData_inputSizeShouldBeCorrect(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

    ds.FromRandom(10, 4)

	for _, input := range ds.Data() {
		if len(input) != 4 {
			t.Fatal("expected input size should be 4, got", len(input))
		}
	}
}

func TestDataset_FromCsv_dataSizeShouldBeCorrect(t *testing.T) {
    ds := ai_backpropagation_continued.NewDataset()

    ds.FromCsv("csv-test.csv", -1, -1, -1)

    if ds.Size() != 10 {
    	t.Fatal("data size should be 10, got", ds.Size())
	}
}

func TestDataset_FromCsv_associationSizeShouldBeCorrect(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", -1, -1, -1)

	for _, input := range ds.Data() {
		if len(input) != 11 {
			t.Fatal("expected input size should be 11, got", len(input))
		}
	}
}

func TestDataset_FromCsv_limitStopsWhenItShould(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", -1, -1, 4)

	if ds.Size() != 4 {
		t.Fatal("data size should be 4, got", ds.Size())
	}
}

func TestDataset_FromCsv_associationSizeShouldBeCorrectWithCustomIndices(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", 1, 2, -1)

	for _, input := range ds.Data() {
		if len(input) != 2 {
			t.Fatal("expected input size should be 2, got", len(input))
		}
	}
}

func TestDataset_FromCsv_associationSizeShouldBeCorrectWithCustomIndicesWhenOnlyOneColumn(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", 0, 0, -1)

	for _, input := range ds.Data() {
		if len(input) != 1 {
			t.Fatal("expected input size should be 1, got", len(input))
		}
	}
}

func TestDataset_FromCsv_normalizationWorks(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", -1, -1, -1).Normalize(0, 255)

	for _, input := range ds.Data() {
		for _, value := range input {
			if value < 0 || value > 1.0 {
				t.Fatal("expected input size should be between 0 and 1.0, got", value)
			}
		}
	}
}

func TestDataset_FromCsv_oneHotEncodingProducesCorrectSizeForSmallNumber(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", 0, 0, 3).OneHotEncode()

	for _, input := range ds.Data() {
		if len(input) != 3 {
			t.Fatal("expected input size of 3, got", len(input))
		}
	}
}

func TestDataset_FromCsv_oneHotEncodingProducesCorrectSizeForLargerNumber(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", 0, 0, -1).OneHotEncode()

	for _, input := range ds.Data() {
		if len(input) != 10 {
			t.Fatal("expected input size of 10, got", len(input))
		}
	}
}

func TestDataset_FromCsv_oneHotEncodingProducesCorrectValues(t *testing.T) {
	ds := ai_backpropagation_continued.NewDataset()

	ds.FromCsv("csv-test.csv", 0, 0, 3).OneHotEncode()

	if ds.Data()[0][0] != 1.0 {
		t.Fatal("expected 1.0, got", ds.Data()[0][0])
	}
	if ds.Data()[0][1] != 0.0 {
		t.Fatal("expected 0.0, got", ds.Data()[0][1])
	}
	if ds.Data()[0][2] != 0.0 {
		t.Fatal("expected 0.0, got", ds.Data()[0][2])
	}
	if ds.Data()[1][0] != 0.0 {
		t.Fatal("expected 0.0, got", ds.Data()[1][0])
	}
	if ds.Data()[1][1] != 1.0 {
		t.Fatal("expected 1.0, got", ds.Data()[1][1])
	}
	if ds.Data()[1][2] != 0.0 {
		t.Fatal("expected 0.0, got", ds.Data()[1][2])
	}
	if ds.Data()[2][0] != 0.0 {
		t.Fatal("expected 0.0, got", ds.Data()[2][0])
	}
	if ds.Data()[2][1] != 0.0 {
		t.Fatal("expected 0.0, got", ds.Data()[2][1])
	}
	if ds.Data()[2][2] != 1.0 {
		t.Fatal("expected 1.0, got", ds.Data()[2][2])
	}
}
