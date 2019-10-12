package runners

import (
	"github.com/gokadin/ai-backpropagation-continued/core"
	"github.com/gokadin/ai-backpropagation-continued/layer"
	"math"
)

func backpropagate(network *core.Network, inputs, expected [][]float64) {
	for i := 0; i < len(inputs); i++ {
		network.Activate(inputs[i])
		calculateDeltas(network, expected[i])
		accumulateGradients(network)
	}
}

func calculateDeltas(network *core.Network, expected []float64) {
	calculateOutputDeltas(network.OutputLayer(), expected)
	calculateHiddenDeltas(network)
}

func calculateHiddenDeltas(network *core.Network) {
	// going backwards from the last hidden layer to the first hidden layer
	for i := network.LayerCount() - 2; i > 0; i-- {
		for _, n := range network.GetLayer(i).Nodes() {
			sumPreviousDeltasAndWeights := 0.0
			for _, c := range n.Connections() {
				sumPreviousDeltasAndWeights += c.NextNode().Delta() * c.Weight()
			}
			n.SetDelta(sumPreviousDeltasAndWeights * network.GetLayer(i).ActivationDerivative()(n.Input()))
		}
	}
}

func calculateOutputDeltas(outputLayer *layer.Layer, expected []float64) {
	for i, n := range outputLayer.Nodes() {
		n.SetDelta(n.Output() - expected[i])
	}
}

func accumulateGradients(network *core.Network) {
	// going backwards from the last hidden layer to the input layer
	for i := len(network.GetLayers()) - 2; i >= 0; i-- {
		for _, node := range network.GetLayer(i).Nodes() {
			for _, connection := range node.Connections() {
				connection.AddGradient(connection.NextNode().Delta() * node.Output())
			}
		}
	}
}

func updateWeights(network *core.Network, batchSize, t int, learningRate, beta1, beta2, epsStable float64) {
	for i := 0; i < len(network.GetLayers()) - 1; i++ {
		for _, node := range network.GetLayer(i).Nodes() {
			for _, c := range node.Connections() {

                /* Adam optimier implementation */

				g := c.GetGradient() / float64(batchSize)

				c.SetVelocity(beta1 * c.GetVelocity() + (1 - beta1) * g)
				c.SetSqrt(beta2 * c.GetSqrt() + (1 - beta2) * math.Pow(g, 2))

				biasCorr := c.GetVelocity() / (1 - math.Pow(beta1, float64(t)))
				sqrtBiasCorr := c.GetSqrt() / (1 - math.Pow(beta2, float64(t)))

				update := learningRate * biasCorr / (math.Sqrt(sqrtBiasCorr) + epsStable)

                c.SetWeight(c.GetWeight() - update)
                c.ResetGradient()
			}
		}
	}
}
