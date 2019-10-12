package core

import (
	"github.com/gokadin/ai-backpropagation-continued"
	"github.com/gokadin/ai-backpropagation-continued/layer"
	"log"
)

type Network struct {
	builder *ai_backpropagation_continued.builder
	layers  []*layer.Layer
}

func NewNetwork() *Network {
	n := &Network{
		layers: make([]*layer.Layer, 0),
	}

	n.builder = ai_backpropagation_continued.newBuilder(n)

	return n
}

func (n *Network) AddInputLayer(size int) *ai_backpropagation_continued.builder {
	n.builder.addInputLayer(size)
	return n.builder
}

func (n *Network) LayerCount() int {
	return len(n.layers)
}

func (n *Network) GetLayers() []*layer.Layer {
	return n.layers
}

func (n *Network) GetLayer(index int) *layer.Layer {
	if index < 0 || index > n.LayerCount() - 1 {
		log.Fatal("requested layer at index", index, "does not exist")
	}

	return n.layers[index]
}

func (n *Network) InputLayer() *layer.Layer {
	if n.LayerCount() == 0 {
		log.Fatal("input layer not set")
	}

	return n.layers[0]
}

func (n *Network) OutputLayer() *layer.Layer {
	if n.LayerCount() == 0 || !n.layers[n.LayerCount() - 1].IsOutputLayer() {
		log.Fatal("output layer not set")
	}

	return n.layers[n.LayerCount() - 1]
}

func (n *Network) Activate(input []float64) {
	n.InputLayer().ResetInputs()
	n.InputLayer().SetInputs(input)
	n.InputLayer().Activate()
}

func (n *Network) ActivateAll(inputs [][]float64) [][]float64 {
	outputs := make([][]float64, len(inputs))
	for i, input := range inputs {
		n.Activate(input)
		outputs[i] = make([]float64, n.OutputLayer().Size())
		for j, outputNode := range n.OutputLayer().Nodes() {
            outputs[i][j] = outputNode.Output()
		}
	}
	return outputs
}
