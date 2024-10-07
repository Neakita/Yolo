using FluentAssertions;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

public sealed class ClassificationTests
{
	private static readonly Rgb24InputProcessor Rgb24InputProcessor = new();

	public ClassificationTests(ITestOutputHelper testOutputHelper)
	{
		_testOutputHelper = testOutputHelper;
	}

	private readonly ITestOutputHelper _testOutputHelper;

	[Theory]
	[InlineData("yolov8n-cls-uint8.onnx", "pizza224.png", "pizza", false)]
	[InlineData("yolov8n-cls-uint8.onnx", "toaster224.png", "toaster", true)]
	[InlineData("yolov8n224fp32cls.onnx", "pizza224.png", "pizza", false)]
	[InlineData("yolov8n224fp32cls.onnx", "toaster224.png", "toaster", true)]
	[InlineData("yolo11n224fp32cls.onnx", "pizza224.png", "pizza", false)]
	[InlineData("yolo11n224fp32cls.onnx", "toaster224.png", "toaster", true)]
	public void ShouldClassifyWhenSizeMatches(
		string modelFileName,
		string imageFileName,
		string expectedClassification,
		bool useGpu)
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, useGpu);
		V8ClassificationProcessor outputProcessor = new();
		var image = TestImageLoader.LoadImage(imageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var classifications = predictor.Predict(imageData.Span2D, Rgb24InputProcessor, outputProcessor);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var actualClassification = predictor.Metadata.ClassesNames[classifications[0].ClassId];
		actualClassification.Should().Be(expectedClassification);
	}
}