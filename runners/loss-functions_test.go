package runners

import (
    "math"
    "testing"
)

func Test_squareError_forOneAssociationReturnsCorrectSize(t *testing.T) {
    outputs := [][]float64{{1.0}}
    expected := [][]float64{{0.0}}

    errors := squareError(outputs, expected)

    if len(errors) != 1 {
        t.Fatalf("expected %d, got %d", 1, len(errors))
    }
}

func Test_squareError_forOneAssociationReturnsCorrectError(t *testing.T) {
    outputs := [][]float64{{1.0}}
    expected := [][]float64{{0.0}}

    errors := squareError(outputs, expected)

    error1 := math.Pow(outputs[0][0] - expected[0][0], 2) / 2
    if errors[0] != error1 {
        t.Fatalf("expected %f, got %f", error1, errors[0])
    }
}

func Test_squareError_forTwoAssociationReturnsCorrectErrorMean(t *testing.T) {
    outputs := [][]float64{{1.0, 1.0}}
    expected := [][]float64{{0.0, 0.5}}

    errors := squareError(outputs, expected)

    error1 := math.Pow(outputs[0][0] - expected[0][0], 2)
    error2 := math.Pow(outputs[0][1] - expected[0][1], 2)
    errorTotal := (error1 + error2) / 2
    if errors[0] != errorTotal {
        t.Fatalf("expected %f, got %f", errorTotal, errors[0])
    }
}
