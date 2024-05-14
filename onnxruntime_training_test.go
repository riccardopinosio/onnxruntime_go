package onnxruntime_go

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path"
	"testing"
)

func loadAnimalsDasets(t *testing.T) map[string][]string {
	t.Helper()
	imagesFiles := map[string][]string{}
	for _, animal := range []string{"dog", "cat", "cow", "elephant"} {
		files, err := os.ReadDir(fmt.Sprintf("test_data/onnxruntime_training_test/processed_images/%s", animal))
		if err != nil {
			t.FailNow()
		}
		for _, f := range files {
			imagesFiles[animal] = append(
				imagesFiles[animal],
				fmt.Sprintf("test_data/onnxruntime_training_test/processed_images/%s/%s", animal, f.Name()))
		}
	}
	return imagesFiles
}

func TestTrainingNotSupported(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Failed creating options: %s\n", e)
		t.FailNow()
	}

	_, e = NewTrainingSession("test_data/onnxruntime_training_test/training_artifacts/checkpoint",
		"test_data/onnxruntime_training_test/training_artifacts/training_model.onnx",
		"test_data/onnxruntime_training_test/training_artifacts/eval_model.onnx",
		"test_data/onnxruntime_training_test/training_artifacts/optimizer_model.onnx",
		nil, nil,
		options)
	if ok := errors.Is(e, trainingNotSupportedError); !ok {
		t.Logf("Creating training session when onnxruntime lib does not support it should return error.")
		t.FailNow()
	}
}

func TestSessionInfo(t *testing.T) {
	t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "test_data/onnxruntime_training_test/onnxruntime_training.so")
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Failed creating options: %s\n", e)
		t.FailNow()
	}
	artifactsPath := path.Join("test_data", "onnxruntime_training_test", "training_artifacts")

	trainingSession, errorSessionCreation := NewTrainingSession(path.Join(artifactsPath, "checkpoint"),
		path.Join(artifactsPath, "training_model.onnx"),
		path.Join(artifactsPath, "eval_model.onnx"),
		path.Join(artifactsPath, "optimizer_model.onnx"),
		nil,
		nil,
		options)

	if errorSessionCreation != nil {
		t.Logf("Failed creating training session: %s\n", errorSessionCreation.Error())
		t.FailNow()
	}
	names, err := trainingSession.GetInputOutputNames()
	if err != nil {
		t.FailNow()
	}
	expectedTrainInputNames := []string{"input", "labels"}
	expectedEvalInputNames := expectedTrainInputNames
	expectedTrainOutputNames := []string{"onnx::loss::2"}
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

// TestBasicTraining tests a basic training flow using the bindings to the C api for on-device onnxruntime training
func TestBasicTraining(t *testing.T) {
	t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "test_data/onnxruntime_training_test/onnxruntime_training.so")
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	trainingArtifactsFolder := path.Join("test_data", "onnxruntime_training_test", "training_artifacts")

	dataset := loadAnimalsDasets(t)
	nSamplePerClass := 20
	epochs := 5
	labelToId := map[string]int{
		"dog":      0,
		"cat":      1,
		"elephant": 2,
		"cow":      3,
	}

	type image [3][224][224]float32

	inputBatchShape := NewShape(4, 3, 224, 224)
	labelBatchShape := NewShape(4)
	inputBatchTensor, err := NewEmptyTensor[float32](inputBatchShape)
	if err != nil {
		t.Fail()
	}
	labelBatchTensor, err := NewEmptyTensor[int64](labelBatchShape)
	if err != nil {
		t.Fail()
	}
	// the loss scalar will hold the loss for each minibatch
	lossScalar, err := NewEmptyScalar[float32]()
	if err != nil {
		t.FailNow()
	}

	trainingSession, errorSessionCreation := NewTrainingSession(
		path.Join(trainingArtifactsFolder, "checkpoint"),
		path.Join(trainingArtifactsFolder, "training_model.onnx"),
		path.Join(trainingArtifactsFolder, "eval_model.onnx"),
		path.Join(trainingArtifactsFolder, "optimizer_model.onnx"),
		[]ArbitraryTensor{inputBatchTensor, labelBatchTensor}, []ArbitraryTensor{lossScalar},
		nil)

	if errorSessionCreation != nil {
		t.Logf("Failed creating training session: %s\n", errorSessionCreation)
		t.FailNow()
	}

	defer func() {
		err := errors.Join(
			inputBatchTensor.Destroy(),
			labelBatchTensor.Destroy(),
			lossScalar.Destroy(),
			trainingSession.Destroy(),
		)
		if err != nil {
			t.Fail()
		}
	}()

	losses := []float32{}

	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float32
		for i := 0; i < nSamplePerClass; i++ {
			inputData := inputBatchTensor.GetData()
			labelData := labelBatchTensor.GetData()

			counter := 0
			counterLabel := 0

			for animal, images := range dataset {
				var imageMap map[string]image
				b, e := os.ReadFile(images[i])
				if e != nil {
					t.FailNow()
				}
				if e := json.Unmarshal(b, &imageMap); e != nil {
					t.FailNow()
				}
				image, ok := imageMap["array"]
				if !ok {
					t.FailNow()
				}
				for _, channel := range image {
					for _, x := range channel {
						for _, y := range x {
							inputData[counter] = y
							counter++
						}
					}
				}
				labelData[counterLabel] = int64(labelToId[animal])
				counterLabel++
			}

			// ort training api - training model execution outputs the training loss and the parameter gradients
			err = trainingSession.TrainStep()
			if err != nil {
				t.FailNow()
			}

			epochLoss += lossScalar.GetData()

			// ort training api - update the model parameters by taking a step in the direction of the gradients
			err = trainingSession.OptimizerStep()
			if err != nil {
				t.FailNow()
			}

			// ort training api - reset the gradients to zero so that new gradients can be computed in the next run
			trainingSession.LazyResetGrad()
		}
		losses = append(losses, epochLoss)
		fmt.Printf("Epoch {%d} Loss {%f}", epoch+1, epochLoss/float32(nSamplePerClass))
	}

	expectedLosses := []float32{20.163158535957336, 8.519233018159866, 4.379776313900948, 2.721357375383377, 1.9140491038560867}

	// check the losses against expected losses from python training example
	// there are slight differences here for now probably due to the order of images, so allow for a 5% deviation tolerance for now
	for i, l := range losses {
		diff := math.Abs(float64(l - expectedLosses[i]))
		deviation := diff / float64(expectedLosses[i])
		if deviation > 0.05 {
			t.FailNow()
		}
	}

	// test the saving of the checkpoint state works
	errSaveCheckpoint := trainingSession.SaveCheckpoint(path.Join("test_data", "onnxruntime_training_test", "finalCheckpoint"), false)
	if errSaveCheckpoint != nil {
		t.Fatalf("Saving of checkpoint failed: %s", errSaveCheckpoint.Error())
	}

	// test the saving of the model works
	errExport := trainingSession.ExportModel(path.Join("test_data", "onnxruntime_training_test", "finalInference.onnx"), []string{"output"})
	if errExport != nil {
		t.Fatalf("Exporting model failed: %s", errExport.Error())
	}
}
