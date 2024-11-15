using System.Runtime.CompilerServices;
using FluentAssertions;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;
using Yolo.Tests.Data;

namespace Yolo.Tests.Helpers;

public sealed class ClassificationTestHelper
{
	public ClassificationTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(string modelFileName, ClassificationTestData data, [CallerMemberName] string testName = "")
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, _useGpu);
		V8ClassificationProcessor outputProcessor = new();
		var image = TestImageLoader.LoadImage(data.ImageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var classifications = predictor.Predict(imageData.Span, Argb32InputProcessor.Instance, outputProcessor);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var actualClassification = predictor.Metadata.ClassesNames[classifications[0].ClassId];
		actualClassification.Should().Be(data.ExpectedClassification);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}