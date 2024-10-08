using System.Runtime.CompilerServices;
using FluentAssertions;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Tests.Helpers;

public sealed class ClassificationTestHelper
{
	public ClassificationTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(string modelFileName, string imageFileName, string expectedClassification, [CallerMemberName] string testName = "")
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, _useGpu);
		V8ClassificationProcessor outputProcessor = new();
		var image = TestImageLoader.LoadImage(imageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var classifications = predictor.Predict(imageData.Span2D, Argb32InputProcessor.Instance, outputProcessor);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var actualClassification = predictor.Metadata.ClassesNames[classifications[0].ClassId];
		actualClassification.Should().Be(expectedClassification);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}