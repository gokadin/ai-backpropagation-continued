package main

import (
	"github.com/gokadin/ann-core/core"
	"github.com/gokadin/ann-core/data"
	"github.com/gokadin/ann-core/layer"
	"github.com/gokadin/ann-core/runners"
	"math/rand"
	"runtime"
	"time"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	network := buildNetwork()

	trainingSet := data.NewDataset()
	trainingSet.FromRandom(10, 8)
	expectedSet := data.NewDataset()
	expectedSet.FromRandom(10, 1).OneHotEncode()
	//trainingSet := data.NewDataset()
	//trainingSet.FromCsv("data/mnist_train_half.csv", 1, -1, -1).Normalize(0, 255)
	//expectedSet := data.NewDataset()
	//expectedSet.FromCsv("data/mnist_train_half.csv", -1, 0, -1).OneHotEncode()

	runner := runners.NewNetworkRunner()
	//runner.SetErrorFunction(runners.ErrorFunctionCrossEntropy)
	runner.SetBatchSize(4)
	runner.SetLearningRate(0.01)
	runner.SetMaxError(0.001)
	runner.SetValidOutputRange(0.05)

	//runner.Train(network, [][]float64{{1, 1}}, [][]float64{{0.5}})
	//runner.Train(network, [][]float64{{1, 0}, {1, 1}, {0, 1}, {0, 0}}, [][]float64{{1}, {0}, {1}, {0}})
	runner.Train(network, trainingSet.Data(), expectedSet.Data())
	runner.Test(network, trainingSet.Data(), expectedSet.Data())
}

func buildNetwork() *core.Network {
	network := core.NewNetwork()
	network.AddInputLayer(8).
		AddHiddenLayer(8, layer.FunctionSigmoid).
		AddOutputLayer(1)

	return network
}