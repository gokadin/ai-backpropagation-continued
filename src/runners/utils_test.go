package runners

import "testing"

func Test_mean(t *testing.T) {
	input := []float64{0.0, 0.5, 1.0}

	result := mean(input)

	if result != 0.5 {
		t.Fatalf("expected %f, got %f", 0.5, result)
	}
}

func Test_partition_sizeWithFittingNumberOfBatchesOnTheFirstIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 0, 2)

	if len(results) != 2 {
		t.Fatalf("expected %d, got %d", 2, len(results))
	}
}

func Test_partition_sizeWithFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 2)

	if len(results) != 2 {
		t.Fatalf("expected %d, got %d", 2, len(results))
	}
}

func Test_partition_sizeWithNonFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 1)

	if len(results) != 1 {
		t.Fatalf("expected %d, got %d", 1, len(results))
	}
}

func Test_partition_dataWithFittingNumberOfBatchesOnTheFirstIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 0, 2)

	if results[0][0] != 1.0 && results[1][0] != 2.0 {
        t.Fatal("data does not match input")
	}
}

func Test_partition_dataWithFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 2)

	if results[0][0] != 3.0 && results[1][0] != 4.0 {
		t.Fatal("data does not match input")
	}
}

func Test_partition_dataWithNonFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 2)

	if results[0][0] != 3.0 {
		t.Fatal("data does not match input")
	}
}
