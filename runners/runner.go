package runners

import (
    "fmt"
	"github.com/gokadin/ai-backpropagation-continued"
	"github.com/gokadin/ai-backpropagation-continued/core"
    "log"
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
    verboseLevel int
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
    nr.validateTrain(network, inputs,  expected)
    fmt.Println("Beginning training of", len(inputs), "associations")

    numBatches := len(inputs) / nr.batchSize
    t := 0
    for epochCounter := 1; epochCounter != nr.epochs; epochCounter++ {
        ai_backpropagation_continued.shuffleDataset(inputs, expected)
        for batchCounter := 0; batchCounter < numBatches; batchCounter++ {
            t++
            batchInputs := ai_backpropagation_continued.partitionData(inputs, batchCounter, nr.batchSize)
            batchExpected := ai_backpropagation_continued.partitionData(expected, batchCounter, nr.batchSize)

            ai_backpropagation_continued.backpropagate(network, batchInputs, batchExpected)
            ai_backpropagation_continued.updateWeights(network, nr.batchSize, t, nr.learningRate, nr.beta1, nr.beta2, nr.epsStable)
        }

        totalOutputs := network.ActivateAll(inputs)
        totalError := ai_backpropagation_continued.mean(ai_backpropagation_continued.squareError(totalOutputs, expected))
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

    //goterm.MoveCursorBackward(100)
    //goterm.Printf("%d/%d   ", batchCounter, batchPerEpoch)
    //progressStr := "[";
    //progressBars := batchCounter * 30 / batchPerEpoch
    //for i := 0; i < progressBars; i++ {
    //    progressStr += "="
    //}
    //for i := 0; i < 30 - progressBars; i++ {
    //    progressStr += "."
    //}
    //progressStr += "]   "
    //goterm.Print(progressStr)
    //goterm.Print("error:", err)
    //goterm.Flush()
}

func (nr *NetworkRunner) Test(network *core.Network, inputs, expected [][]float64) {
    nr.validateTest(network, inputs, expected)

	outputs := network.ActivateAll(inputs)
	err := ai_backpropagation_continued.mean(ai_backpropagation_continued.squareError(outputs, expected))

	fmt.Println("Error:", err)
}

func (nr *NetworkRunner) validateTest(network *core.Network, inputs, expected [][]float64) {

}

func (nr *NetworkRunner) validateTrain(network *core.Network, inputs, expected [][]float64) {
    if nr.batchSize == 0 || nr.batchSize > len(inputs) {
        log.Fatal("batch size is incompatible with the given training set")
    }
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

func (nr *NetworkRunner) SetVerboseLevel(verboseLevel int) {
    nr.verboseLevel = verboseLevel
}
