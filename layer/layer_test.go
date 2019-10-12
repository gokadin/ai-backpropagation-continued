package layer

import (
	"github.com/gokadin/ai-backpropagation-continued"
	"testing"
)

const someLayerSize = 10

func Test_layer_initializes_with_correct_size(t *testing.T) {
	layer := ai_backpropagation_continued.NewLayer(someLayerSize, ai_backpropagation_continued.FunctionIdentity)

	if len(layer.nodes) != someLayerSize + 1 {
		t.Fatalf("Layer initialized with wrong size. Expected %d, got %d", someLayerSize + 1, len(layer.nodes))
	}
}
