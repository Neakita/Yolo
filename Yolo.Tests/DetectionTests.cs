using System.Runtime.CompilerServices;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;
using Yolo.Tests.Helpers;

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
	[InlineData("yolov10n160fp32.onnx", "bus160.png", BusExpectedDetections, false)]
	[InlineData("yolov10n160fp32.onnx", "bus160.png", BusExpectedDetections, true)]
	[InlineData("yolov10n224fp32.onnx", "bus224.png", BusExpectedDetections, false)]
	[InlineData("yolov10n224fp32.onnx", "bus224.png", BusExpectedDetections, true)]
	[InlineData("yolov10n320fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov10n320fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov10n480fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov10n480fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov10n640fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov10n640fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov10n800fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov10n800fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	public void ShouldDetectWhenSizeMatches(
		string modelFileName,
		string imageFileName,
		string expectedDetections,
		bool useGpu)
	{
		PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections, useGpu);
	}

	[Theory]
	[InlineData("yolov8n160fp32.onnx", "bus224.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus224.png", BusExpectedDetections, true)]
	[InlineData("yolov8n160fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov8n160fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov8n160fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n160fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov8n224fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov8n224fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov8n224fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov8n224fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov8n224fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n224fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n224fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n224fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov8n320fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov8n320fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov8n320fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n320fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n320fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n320fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov8n480fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n480fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n480fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n480fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov8n640fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n640fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov10n160fp32.onnx", "bus224.png", BusExpectedDetections, false)]
	[InlineData("yolov10n160fp32.onnx", "bus224.png", BusExpectedDetections, true)]
	[InlineData("yolov10n160fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov10n160fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov10n160fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov10n160fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov10n160fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov10n160fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov10n160fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov10n160fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov10n224fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov10n224fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov10n224fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov10n224fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov10n224fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov10n224fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov10n224fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov10n224fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov10n320fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov10n320fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov10n320fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov10n320fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov10n320fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov10n320fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov10n480fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov10n480fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov10n480fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov10n480fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	[InlineData("yolov10n640fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov10n640fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	public void ShouldDetectWhenImageSizeIsHigher(
		string modelFileName,
		string imageFileName,
		string expectedDetections,
		bool useGpu)
	{
		PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections, useGpu);
	}

	private void PredictPlotAndAssert(string modelFileName, string imageFileName, string expectedDetections, bool useGpu, [CallerMemberName] string testName = "")
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, useGpu);
		OutputProcessor<Detection> outputProcessor = predictor.Metadata.ModelVersion switch
		{
			8 => new V8DetectionProcessor(predictor.Metadata),
			10 => new V10DetectionProcessor(predictor.Metadata),
			_ => throw new ArgumentOutOfRangeException()
		};
		outputProcessor.MinimumConfidence = 0.5f;
		var image = TestImageLoader.LoadImage(imageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var detections = predictor.Predict(imageData.Span2D, Rgb24InputProcessor, outputProcessor);
		DetectionsOutputHelper.WriteDetections(_testOutputHelper, predictor.Metadata, detections);
		var plotted = DetectionsPlottingHelper.Plot(image, predictor.Metadata, detections);
		ImageSaver.Save(plotted, modelFileName, imageFileName, useGpu, testName);
		DetectionAssertionHelper.AssertPrediction(predictor.Metadata, expectedDetections, detections);
	}
}