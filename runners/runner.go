package runners

import (
    "fmt"
    "github.com/gokadin/ai-backpropagation-continued/algorithm"
    "github.com/gokadin/ai-backpropagation-continued/core"
)

type NetworkRunner struct {
    learningRate float64
    beta1 float64
    beta2 float64
    epsStable float64
    batchSize int
    epochs int
    maxError float64
    validOutputRange float64
}

func NewNetworkRunner() *NetworkRunner {
    return &NetworkRunner{
        learningRate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsStable: 1e-08,
        batchSize: 1,
        epochs: -1,
        maxError: 0.001,
        validOutputRange: 0.1,
    }
}

func (nr *NetworkRunner) Train(network *core.Network, inputs, expected [][]float64) {
    fmt.Println("Beginning training of", len(inputs), "associations")

    numBatches := len(inputs) / nr.batchSize
    t := 0
    for epochCounter := 1; epochCounter != nr.epochs; epochCounter++ {
        shuffleDataset(inputs, expected)
        for batchCounter := 0; batchCounter < numBatches; batchCounter++ {
            t++
            batchInputs := partitionData(inputs, batchCounter, nr.batchSize)
            batchExpected := partitionData(expected, batchCounter, nr.batchSize)

            algorithm.Backpropagate(network, batchInputs, batchExpected)
            algorithm.UpdateWeights(network, nr.batchSize, t, nr.learningRate, nr.beta1, nr.beta2, nr.epsStable)
        }

        totalOutputs := network.ActivateAll(inputs)
        totalError := mean(squareError(totalOutputs, expected))
        if epochCounter % 1 == 0 {
            nr.logEpoch(epochCounter, totalError)
        }
        if totalError <= nr.maxError {
            fmt.Println("Training finished in", epochCounter, "with error", totalError)
            return
        }
    }
}

func (nr NetworkRunner) logEpoch(epochCounter int, totalError float64) {
    fmt.Println("Epoch", epochCounter, "finished with error", totalError)
}

func (nr NetworkRunner) logBatchProgress(batchCounter, batchPerEpoch int, err float64) {
	fmt.Printf("%d/%d   error: %f\n", batchCounter, batchPerEpoch, err)
}

func (nr *NetworkRunner) Test(network *core.Network, inputs, expected [][]float64) {
	outputs := network.ActivateAll(inputs)
	err := mean(squareError(outputs, expected))

	fmt.Println("Error:", err)
}

func (nr *NetworkRunner) SetBatchSize(batchSize int) {
    nr.batchSize = batchSize
}

func (nr *NetworkRunner) SetEpochLimit(epochs int) {
    nr.epochs = epochs
}

func (nr *NetworkRunner) SetLearningRate(learningRate float64) {
    nr.learningRate = learningRate
}

func (nr *NetworkRunner) SetMaxError(maxError float64) {
    nr.maxError = maxError
}

func (nr *NetworkRunner) SetValidOutputRange(validOutputRange float64) {
    nr.validOutputRange = validOutputRange
}
