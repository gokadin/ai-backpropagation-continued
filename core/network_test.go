package core

import (
	"github.com/gokadin/ai-backpropagation-continued/layer"
	"github.com/stretchr/testify/assert"
	"math"
	"math/rand"
	"testing"
)

const floatDelta = 0.000001

func buildSimpleTestNetwork(inputCount, hiddenCount, outputCount int, activationFunction string) *Network {
	network := NewNetwork()
	network.AddInputLayer(inputCount).
		AddHiddenLayer(hiddenCount, activationFunction).
		AddOutputLayer(outputCount, layer.FunctionIdentity)

	return network
}

func generateSimpleData(inputCount, outputCount, associations int) ([][]float64, [][]float64) {
	inputs := make([][]float64, associations)
	for i := 0; i < associations; i++ {
		input := make([]float64, inputCount)
		for j := 0; j < inputCount; j++ {
			input[j] = rand.Float64()
		}
		inputs[i] = input
	}

	outputs := make([][]float64, associations)
	for i := 0; i < associations; i++ {
		output := make([]float64, inputCount)
		for j := 0; j < outputCount; j++ {
			output[j] = rand.Float64()
		}
		outputs[i] = output
	}

	return inputs, outputs
}

func Test_forwardPass_setsCorrectInputValues(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.InputLayer().SetInputs(inputs[0])

	for i, value := range inputs[0] {
		assert.Equal(t, value, net.InputLayer().Nodes()[i].Input())
	}
}

func Test_forwardPass_outputIsCorrect(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	expected := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight() + net.InputLayer().Bias().Connections()[0].Weight()
	expected = expected * net.GetLayer(1).Nodes()[0].Connection(0).Weight() + net.GetLayer(1).Bias().Connections()[0].Weight()
	assert.InDelta(t, expected, net.OutputLayer().Nodes()[0].Output(), floatDelta)
}

func Test_forwardPass_outputIsCorrectWithSigmoidActivationHiddenNode(t *testing.T) {
	net := buildSimpleTestNetwork(1, 1, 1, layer.FunctionSigmoid)
	inputs, _ := generateSimpleData(1, 1, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	expected := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight() + net.InputLayer().Bias().Connections()[0].Weight()
	expected = 1 / (1 + math.Pow(math.E, -expected))
	expected = expected * net.GetLayer(1).Nodes()[0].Connection(0).Weight() + net.GetLayer(1).Bias().Connections()[0].Weight()
	assert.InDelta(t, expected, net.OutputLayer().Nodes()[0].Output(), floatDelta)
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 1, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 1, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	h1 := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.InputLayer().Nodes()[1].Connection(0).Weight()
	h1 += net.InputLayer().Bias().Connections()[0].Weight()
	expected := h1 * net.GetLayer(1).Nodes()[0].Connection(0).Weight() + net.GetLayer(1).Bias().Connections()[0].Weight()
	assert.InDelta(t, expected, net.OutputLayer().Nodes()[0].Output(), floatDelta)
}

func Test_forwardPass_outputIsCorrectWithTwoInputNodesAndTwoHiddenNodes(t *testing.T) {
	net := buildSimpleTestNetwork(2, 2, 1, layer.FunctionIdentity)
	inputs, _ := generateSimpleData(2, 2, 1)

	net.InputLayer().ResetInputs()
	net.InputLayer().SetInputs(inputs[0])
	net.InputLayer().Activate()

	h1 := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(0).Weight()
	h1 += inputs[0][1] * net.InputLayer().Nodes()[1].Connection(0).Weight()
	h1 += net.InputLayer().Bias().Connections()[0].Weight()
	h2 := inputs[0][0] * net.InputLayer().Nodes()[0].Connection(1).Weight()
	h2 += inputs[0][1] * net.InputLayer().Nodes()[1].Connection(1).Weight()
	h2 += net.InputLayer().Bias().Connections()[1].Weight()
	expected := h1 * net.GetLayer(1).Nodes()[0].Connection(0).Weight()
	expected += h2 * net.GetLayer(1).Nodes()[1].Connection(0).Weight()
	expected += net.GetLayer(1).Bias().Connections()[0].Weight()
	assert.InDelta(t, expected, net.OutputLayer().Nodes()[0].Output(), floatDelta)
}
