package layer

import (
	"github.com/gokadin/ai-backpropagation-continued/node"
	"log"
	"math"
	"math/rand"
)

const defaultBiasValue = 1.0

type Layer struct {
	nodes     []*node.Node
	bias *node.Node
	nextLayer *Layer
    activationFunctionName string
	isOutputLayer bool
}

func NewLayer(size int, activationFunctionName string) *Layer {
	return &Layer{
		nodes: initializeNodes(size),
		bias: node.NewBiasNode(defaultBiasValue),
        activationFunctionName: activationFunctionName,
		isOutputLayer: false,
	}
}

func NewOutputLayer(size int, activationFunctionName string) *Layer {
	layer := NewLayer(size, activationFunctionName)
	layer.isOutputLayer = true
	return layer
}

func initializeNodes(size int) []*node.Node {
	nodes := make([]*node.Node, size)
	for i := range nodes {
		nodes[i] = node.NewNode()
	}
	return nodes
}

func (l *Layer) Size() int {
	return len(l.nodes)
}

func (l *Layer) IsOutputLayer() bool {
	return l.isOutputLayer
}

func (l *Layer) ConnectTo(nextLayer *Layer) {
	l.nextLayer = nextLayer

	for _, n := range l.nodes {
		for _, nextNode := range nextLayer.nodes {

			/* Better weight initialization */

			weight := rand.NormFloat64() / math.Sqrt(float64(l.Size()))
			n.ConnectTo(nextNode, weight)
		}
	}

	for _, nextNode := range nextLayer.nodes {
		weight := rand.NormFloat64() / math.Sqrt(float64(l.Size()))
		l.bias.ConnectTo(nextNode, weight)
	}
}

func (l *Layer) Nodes() []*node.Node {
	return l.nodes
}

func (l *Layer) Node(index int) *node.Node {
	return l.nodes[index]
}

func (l *Layer) Bias() *node.Node {
	return l.bias
}

func (l *Layer) Parameters() []*node.Node {
    if l.isOutputLayer {
    	return l.nodes
	}

    return append(l.nodes, l.bias)
}

func (l *Layer) SetInputs(values []float64) {
	if len(values) != l.Size() {
		log.Fatal("Cannot set values, size mismatch:", len(values), "!=", l.Size())
	}

	for i, value := range values {
		l.nodes[i].SetInput(value)
	}
}

func (l *Layer) ResetInputs() {
	for _, n := range l.nodes {
		n.ResetInput()
	}

	if l.nextLayer != nil {
		l.nextLayer.ResetInputs()
	}
}

func (l *Layer) Activate() {
	switch l.activationFunctionName {
	case FunctionSoftmax:
		l.activateSoftmax()
		break
	default:
		for _, n := range l.nodes {
			n.Activate(getActivationFunction(l.activationFunctionName))
		}
		l.bias.Activate(nil)
		break
	}

	if l.nextLayer != nil {
		l.nextLayer.Activate()
	}
}

func (l *Layer) ActivationDerivative() func (x float64) float64 {
	return getActivationFunctionDerivative(l.activationFunctionName)
}

// INCORRECT
func (l *Layer) activateSoftmax() {
	sum := 0.0
	for _, n := range l.nodes {
		sum += math.Pow(math.E, n.Input())
	}
	for _, n := range l.nodes {
		inputExp := math.Pow(math.E, n.Input())
		partialSum := sum - inputExp
		n.SetOutput(inputExp / partialSum)
	}
}
