package main

import (
	"github.com/gokadin/ann-core/core"
	"github.com/gokadin/ann-core/data"
	"github.com/gokadin/ann-core/layer"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	network := buildNetwork()
	dataset := buildDataset()

	_ = network
	_ = dataset

	// UNDER CONSTRUCTION

	//network.Run(dataset.TrainingSet())
}

func buildNetwork() *core.Network {
	network := core.NewNetwork()
	network.AddInputLayer(768).
		AddHiddenLayer(768, layer.FunctionSigmoid).
		AddOutputLayer(10)

	return network
}

func buildDataset() *data.Dataset {
	dataset := data.NewDataset()
	dataset.ReadCsvAsTraining("data/mnist_train_half.csv")
	dataset.ReadCsvAsTest("data/mnist_test_full.csv")

	return dataset
}
