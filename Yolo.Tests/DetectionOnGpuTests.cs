using Xunit.Abstractions;

namespace Yolo.Tests;

[Collection("gpu")]
public sealed class DetectionOnGpuTests
{
	public DetectionOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_detectionTestHelper = new DetectionTestHelper(testOutputHelper, true);
	}

	[Theory]
	[InlineData("yolov8n-uint8.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n160fp32.onnx", "bus160.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n224fp32.onnx", "bus224.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n480fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n640fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n800fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n160fp32.onnx", "bus160.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n224fp32.onnx", "bus224.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n320fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n480fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n640fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n800fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n160fp32.onnx", "bus160.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n224fp32.onnx", "bus224.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n320fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n480fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
    [InlineData("yolo11n640fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n800fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	public void ShouldDetectOnGpuWhenSizeMatches(
		string modelFileName,
		string imageFileName,
		string expectedDetections)
	{
		_detectionTestHelper.PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections);
	}

	[Theory]
	[InlineData("yolov8n160fp32.onnx", "bus224.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n160fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n160fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n160fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n160fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n224fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n224fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n224fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n224fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n320fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n320fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n320fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n480fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n480fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n640fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n160fp32.onnx", "bus224.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n160fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n160fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n160fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n160fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n224fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n224fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n224fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n224fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n320fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n320fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n320fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n480fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n480fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolov10n640fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n160fp32.onnx", "bus224.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n160fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n160fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n160fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n160fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n224fp32.onnx", "bus320.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n224fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n224fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n224fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n320fp32.onnx", "bus480.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n320fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n320fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n480fp32.onnx", "bus640.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n480fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n640fp32.onnx", "bus800.png", DetectionTestHelper.BusExpectedPrediction)]
	public void ShouldDetectOnGpuWhenImageSizeIsHigher(
		string modelFileName,
		string imageFileName,
		string expectedDetections)
	{
		_detectionTestHelper.PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections);
	}

	private readonly DetectionTestHelper _detectionTestHelper;
}