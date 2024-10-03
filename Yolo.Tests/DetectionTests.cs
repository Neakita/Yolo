using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Tests;

public class DetectionTests
{
	// 2 persons are clearly visible and 2 more are only partially.
	// bus is huge object on the image and locates in the center
	// the sign is barely noticeable
	private const string BusExpectedDetections = "person:2-4,bus:1,stop sign:0-1";
	private static readonly Rgb24InputProcessor Rgb24InputProcessor = new();

	public DetectionTests(ITestOutputHelper testOutputHelper)
	{
		_testOutputHelper = testOutputHelper;
	}

	private readonly ITestOutputHelper _testOutputHelper;

	[Theory]
	[InlineData("yolov8n-uint8.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n-uint8.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n160fp32.onnx", "bus160.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus160.png", BusExpectedDetections, true)]
	[InlineData("yolov8n224fp32.onnx", "bus224.png", BusExpectedDetections, false)]
	[InlineData("yolov8n224fp32.onnx", "bus224.png", BusExpectedDetections, true)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov8n480fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov8n480fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov8n640fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n640fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n800fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n800fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	public void ShouldDetectFromImageOfSameSize(
		string modelFileName,
		string imageFileName,
		string expectedDetections,
		bool useGpu)
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, useGpu);
		V8DetectionProcessor outputProcessor = new(predictor.Metadata);
		var image = TestImageLoader.LoadImage(imageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var detections = predictor.Predict(imageData.Span2D, Rgb24InputProcessor, outputProcessor);
		DetectionAssertionHelper.AssertPrediction(predictor.Metadata, expectedDetections, detections);
		DetectionsOutputHelper.WriteDetections(_testOutputHelper, predictor.Metadata, detections);
		var plotted = DetectionsPlottingHelper.Plot(image, predictor.Metadata, detections);
		ImageSaver.Save(plotted, modelFileName, imageFileName, useGpu);
	}
}