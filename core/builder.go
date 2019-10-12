package core

import (
	"github.com/gokadin/ai-backpropagation-continued"
	"github.com/gokadin/ai-backpropagation-continued/layer"
	"log"
)

type builder struct {
	network *ai_backpropagation_continued.Network
}

func newBuilder(network *ai_backpropagation_continued.Network) *builder {
	return &builder{
		network: network,
	}
}

func (b *builder) addInputLayer(size int) {
	if len(b.network.layers) != 0 {
		log.Fatal("You cannot add an input layer after adding other layers.")
	}

	b.network.layers = append(b.network.layers, layer.NewLayer(size, layer.FunctionIdentity))
}

func (b *builder) AddHiddenLayer(size int, activationFunctionName string) *builder {
	if len(b.network.layers) == 0 {
		log.Fatal("You must add an input layer before adding a hidden layer.")
	}

	if activationFunctionName == layer.FunctionSoftmax {
		log.Fatal("Only the output layer can be of type softmax")
	}

	b.addLayer(layer.NewLayer(size, activationFunctionName))
	return b
}

func (b *builder) AddOutputLayer(size int, activationFunctionName string) *builder {
	if len(b.network.layers) < 2 {
		log.Fatal("You must add an input layer and at least one hidden layer before adding an output layer.")
	}

	b.addLayer(layer.NewOutputLayer(size, activationFunctionName))
	return b
}

func (b *builder) addLayer(l *layer.Layer) {
	b.network.layers[len(b.network.layers) - 1].ConnectTo(l)
	b.network.layers = append(b.network.layers, l)
}
