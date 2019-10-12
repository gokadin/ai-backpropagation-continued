package data

type Dataset struct {
	builder *builder
    data [][]float64
}

func NewDataset() *Dataset {
	dataset := &Dataset{
        data: make([][]float64, 0),
	}

	dataset.builder = newBuilder(dataset)
	return dataset
}

func (d *Dataset) FromCsv(filename string, startIndex, endIndex, limit int) *builder {
	return d.builder.readCsv(filename, startIndex, endIndex, limit)
}

func (d *Dataset) FromRandom(associations, size int) *builder {
	return d.builder.readRandom(associations, size)
}

func (d *Dataset) Data() [][]float64 {
	return d.data
}

func (d *Dataset) Size() int {
	return len(d.data)
}
