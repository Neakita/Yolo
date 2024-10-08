using Xunit.Abstractions;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

[Collection("gpu")]
public sealed class PoseOnGpuTests
{
	public PoseOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_poseTestHelper = new PoseTestHelper(testOutputHelper, true);
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
	public void ShouldPoseOnGpuWhenSizeMatches(
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
	public void ShouldPoseOnGpuWhenImageSizeIsHigher(
		string modelFileName,
		string imageFileName,
		string expectedDetections)
	{
		_poseTestHelper.PredictPlotAndAssert(modelFileName, imageFileName, expectedDetections);
	}

	private readonly PoseTestHelper _poseTestHelper;
}