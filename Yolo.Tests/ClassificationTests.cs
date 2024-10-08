using Xunit.Abstractions;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

public sealed class ClassificationTests
{
	public ClassificationTests(ITestOutputHelper testOutputHelper)
	{
		_testHelper = new ClassificationTestHelper(testOutputHelper, false);
	}

	[Theory]
	[InlineData("yolov8n-cls-uint8.onnx", "pizza224.png", "pizza")]
	[InlineData("yolov8n224fp32cls.onnx", "pizza224.png", "pizza")]
	[InlineData("yolo11n224fp32cls.onnx", "pizza224.png", "pizza")]
	[InlineData("yolov8n-cls-uint8.onnx", "toaster224.png", "toaster")]
	[InlineData("yolov8n224fp32cls.onnx", "toaster224.png", "toaster")]
	[InlineData("yolo11n224fp32cls.onnx", "toaster224.png", "toaster")]
	public void ShouldClassifyWhenSizeMatches(
		string modelFileName,
		string imageFileName,
		string expectedClassification)
	{
		_testHelper.PredictPlotAndAssert(modelFileName, imageFileName, expectedClassification);
	}

	private readonly ClassificationTestHelper _testHelper;
}