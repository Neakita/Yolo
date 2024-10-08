using Xunit.Abstractions;

namespace Yolo.Tests;

public sealed class PoseTests
{
	public PoseTests(ITestOutputHelper testOutputHelper)
	{
		_poseTestHelper = new PoseTestHelper(testOutputHelper, false);
	}

	[Theory]
	[InlineData("yolov8n-pose-uint8.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose160fp32.onnx", "bus160.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose224fp32.onnx", "bus224.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose320fp32.onnx", "bus320.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose480fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose640fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose800fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose160fp32.onnx", "bus160.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose224fp32.onnx", "bus224.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose320fp32.onnx", "bus320.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose480fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose640fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose800fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	public void ShouldPoseWhenSizeMatches(
		string modelFileName,
		string imageFileName,
		string expectedDetections)
	{
		_poseTestHelper.PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections);
	}

	[Theory]
	[InlineData("yolov8n-pose160fp32.onnx", "bus224.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose160fp32.onnx", "bus320.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose160fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose160fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose160fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose224fp32.onnx", "bus320.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose224fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose224fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose224fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose320fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose320fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose320fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose480fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose480fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolov8n-pose640fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose160fp32.onnx", "bus224.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose160fp32.onnx", "bus320.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose160fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose160fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose160fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose224fp32.onnx", "bus320.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose224fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose224fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose224fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose320fp32.onnx", "bus480.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose320fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose320fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose480fp32.onnx", "bus640.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose480fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	[InlineData("yolo11n-pose640fp32.onnx", "bus800.png", PoseTestHelper.BusExpectedPrediction)]
	public void ShouldPoseWhenImageSizeIsHigher(
		string modelFileName,
		string imageFileName,
		string expectedDetections)
	{
		_poseTestHelper.PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections);
	}

	private readonly PoseTestHelper _poseTestHelper;
}