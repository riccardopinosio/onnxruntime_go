package onnxruntime_go

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"path"
	"testing"
)

func TestTrainingNotSupported(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Failed creating options: %s\n", e)
		t.FailNow()
	}

	trainingSession, e := NewTrainingSession("test_data/onnxruntime_training_test/training_artifacts/checkpoint",
		"test_data/onnxruntime_training_test/training_artifacts/training_model.onnx",
		"test_data/onnxruntime_training_test/training_artifacts/eval_model.onnx",
		"test_data/onnxruntime_training_test/training_artifacts/optimizer_model.onnx",
		nil, nil,
		options)
	if ok := errors.Is(e, trainingNotSupportedError); !ok {
		t.Logf("Creating training session when onnxruntime lib does not support it should return error.")
		t.FailNow()
	}
	defer func(session *TrainingSession) {
		err := errors.Join(
			trainingSession.Destroy(),
		)
		if err != nil {
			t.Fail()
		}
	}(trainingSession)
}

func TestSessionInfo(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	if !IsTrainingSupported() {
		t.Skipf("Training is not supported on this platform/onnxruntime build.")
	}

	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Failed creating options: %s\n", e)
		t.FailNow()
	}
	artifactsPath := path.Join("test_data", "training_test")

	trainingSession, errorSessionCreation := NewTrainingSession(path.Join(artifactsPath, "checkpoint"),
		path.Join(artifactsPath, "training_model.onnx"),
		path.Join(artifactsPath, "eval_model.onnx"),
		path.Join(artifactsPath, "optimizer_model.onnx"),
		nil,
		nil,
		options)

	defer func(session *TrainingSession) {
		err := errors.Join(
			trainingSession.Destroy(),
		)
		if err != nil {
			t.Fail()
		}
	}(trainingSession)

	if errorSessionCreation != nil {
		t.Logf("Failed creating training session: %s\n", errorSessionCreation.Error())
		t.FailNow()
	}
	names, err := trainingSession.GetInputOutputNames()
	if err != nil {
		t.FailNow()
	}
	expectedTrainInputNames := []string{"input", "target"}
	expectedEvalInputNames := expectedTrainInputNames
	expectedTrainOutputNames := []string{"onnx::reducemean_output::5"}
	expectedEvalOutputNames := expectedTrainOutputNames

	for i, v := range names.TrainingInputNames {
		if v != expectedTrainInputNames[i] {
			t.FailNow()
		}
	}
	for i, v := range names.TrainingOutputNames {
		if v != expectedTrainOutputNames[i] {
			t.FailNow()
		}
	}
	for i, v := range names.EvalInputNames {
		if v != expectedEvalInputNames[i] {
			t.FailNow()
		}
	}
	for i, v := range names.EvalOutputNames {
		if v != expectedEvalOutputNames[i] {
			t.FailNow()
		}
	}
}

func generateBatchData(nBatches int, batchSize int) map[int]map[string][]float32 {
	batchData := map[int]map[string][]float32{}

	source := rand.NewSource(1234)
	g := rand.New(source)

	for i := 0; i < nBatches; i++ {
		inputCounter := 0
		outputCounter := 0
		inputSlice := make([]float32, batchSize*4)
		outputSlice := make([]float32, batchSize*2)
		batchData[i] = map[string][]float32{}

		// generate random data for batch
		for n := 0; n < batchSize; n++ {
			var sum float32
			min := float32(1)
			max := float32(-1)
			for i := 0; i < 4; i++ {
				r := g.Float32()
				inputSlice[inputCounter] = r
				inputCounter++
				if r > max {
					max = r
				}
				if r < min {
					min = r
				}
				sum = sum + r
			}
			outputSlice[outputCounter] = sum
			outputSlice[outputCounter+1] = max - min
			outputCounter = outputCounter + 2
		}
		batchData[i]["input"] = inputSlice
		batchData[i]["output"] = outputSlice
	}
	return batchData
}

// TestTraining tests a basic training flow using the bindings to the C api for on-device onnxruntime training
func TestTraining(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	if !IsTrainingSupported() {
		t.Skipf("Training is not supported on this platform/onnxruntime build.")
	}

	trainingArtifactsFolder := path.Join("test_data", "training_test")

	// generate training data
	batchSize := 10
	nBatches := 10

	// holds inputs/outputs and loss for each training batch
	batchInputShape := NewShape(int64(batchSize), 1, 4)
	batchTargetShape := NewShape(int64(batchSize), 1, 2)
	batchInputTensor, err := NewEmptyTensor[float32](batchInputShape)
	if err != nil {
		t.Fail()
	}
	batchTargetTensor, err := NewEmptyTensor[float32](batchTargetShape)
	if err != nil {
		t.Fail()
	}
	lossScalar, err := NewEmptyScalar[float32]()
	if err != nil {
		t.FailNow()
	}

	trainingSession, errorSessionCreation := NewTrainingSession(
		path.Join(trainingArtifactsFolder, "checkpoint"),
		path.Join(trainingArtifactsFolder, "training_model.onnx"),
		path.Join(trainingArtifactsFolder, "eval_model.onnx"),
		path.Join(trainingArtifactsFolder, "optimizer_model.onnx"),
		[]ArbitraryTensor{batchInputTensor, batchTargetTensor}, []ArbitraryTensor{lossScalar},
		nil)

	if errorSessionCreation != nil {
		t.Logf("Failed creating training session: %s\n", errorSessionCreation)
		t.FailNow()
	}

	// cleanup after test run
	defer func(session *TrainingSession, tensors []ArbitraryTensor) {
		var errs []error
		errs = append(errs, session.Destroy())
		for _, t := range tensors {
			errs = append(errs, t.Destroy())
		}
		if errors.Join(errs...) != nil {
			t.Fail()
		}
	}(trainingSession, []ArbitraryTensor{batchInputTensor, batchTargetTensor, lossScalar})

	losses := []float32{}
	epochs := 100
	batchData := generateBatchData(nBatches, batchSize)

	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float32 // total epoch loss

		for i := 0; i < nBatches; i++ {
			inputSlice := batchInputTensor.GetData()
			outputSlice := batchTargetTensor.GetData()

			copy(inputSlice, batchData[i]["input"])
			copy(outputSlice, batchData[i]["output"])

			// train on batch
			err = trainingSession.TrainStep()
			if err != nil {
				t.Fatalf(err.Error())
			}

			epochLoss = epochLoss + lossScalar.GetData()

			err = trainingSession.OptimizerStep()
			if err != nil {
				t.FailNow()
			}

			// ort training api - reset the gradients to zero so that new gradients can be computed for next batch
			trainingSession.LazyResetGrad()
		}
		if epoch%10 == 0 {
			fmt.Printf("Epoch {%d} Loss {%f}\n", epoch+1, epochLoss/float32(batchSize*nBatches))
			losses = append(losses, epochLoss/float32(batchSize*nBatches))
		}
	}

	expectedLosses := []float32{
		0.119080305,
		0.07214756,
		0.03532303,
		0.020265112,
		0.018456006,
		0.016774517,
		0.015200602,
		0.013888882,
		0.012684849,
		0.011486121,
	}

	for i, l := range losses {
		diff := math.Abs(float64(l - expectedLosses[i]))
		deviation := diff / float64(expectedLosses[i])
		if deviation > 0.6 {
			t.FailNow()
		}
	}

	// test the saving of the checkpoint state
	errSaveCheckpoint := trainingSession.SaveCheckpoint(path.Join("test_data", "training_test", "finalCheckpoint"), false)
	if errSaveCheckpoint != nil {
		t.Fatalf("Saving of checkpoint failed: %s", errSaveCheckpoint.Error())
	}

	// test the saving of the model
	errExport := trainingSession.ExportModel(path.Join("test_data", "training_test", "final_inference.onnx"), []string{"output"})
	if errExport != nil {
		t.Fatalf("Exporting model failed: %s", errExport.Error())
	}

	// load the model back in and test in-sample predictions for the first batch
	// (we care about correctness more than generalization here)
	session, err := NewAdvancedSession(path.Join("test_data", "training_test", "final_inference.onnx"),
		[]string{"input"}, []string{"output"},
		[]ArbitraryTensor{batchInputTensor}, []ArbitraryTensor{batchTargetTensor}, nil)
	if err != nil {
		t.FailNow()
	}

	defer func(s *AdvancedSession) {
		err := s.Destroy()
		if err != nil {
			t.FailNow()
		}
	}(session)

	// Calling Run() will run the network, reading the current contents of the
	// input tensors and modifying the contents of the output tensors.
	copy(batchInputTensor.GetData(), batchData[0]["input"])
	err = session.Run()
	if err != nil {
		t.FailNow()
	}

	expectedOutput := []float32{
		2.4524384,
		0.65120333,
		2.5457804,
		0.6102175,
		1.6276635,
		0.573755,
		1.7900972,
		0.59951085,
		3.1650176,
		0.66626525,
		1.9361509,
		0.571084,
		2.0798547,
		0.6060241,
		0.9611889,
		0.52100605,
		1.4070896,
		0.5412475,
		2.1449144,
		0.5985652,
	}

	for i, l := range batchTargetTensor.GetData() {
		diff := math.Abs(float64(l - expectedOutput[i]))
		deviation := diff / float64(expectedOutput[i])
		if deviation > 0.6 {
			t.FailNow()
		}
	}
}
