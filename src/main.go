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
	//trainingSet.FromCsv("data/mnist_train_half.csv", 1, -1, -1).Normalize(0, 255)
	trainingSet.FromRandom(10, 4)
	expectedSet := data.NewDataset()
	expectedSet.FromRandom(10, 1)
	//expectedSet.FromCsv("data/mnist_train_half.csv", 0, 0, -1).OneHotEncode()

	runner := runners.NewNetworkRunner()
	runner.SetBatchSize(2)
	runner.SetLearningRate(0.01)
	runner.SetMaxError(0.0001)
	runner.SetValidOutputRange(0.05)
	//runner.SetVerboseLevel(runners.VerboseLevelLow)

	runner.Train(network, trainingSet.Data(), expectedSet.Data())
	//runner.Test(network, trainingSet.Data(), expectedSet.Data())
}

func buildNetwork() *core.Network {
	network := core.NewNetwork()
	network.AddInputLayer(4).
	//network.AddInputLayer(784).
		AddHiddenLayer(4, layer.FunctionSigmoid).
		//AddHiddenLayer(784, layer.FunctionSigmoid).
		//AddOutputLayer(10)
		AddOutputLayer(1)

	return network
}
