using FluentAssertions;
using Yolo.ImageSharp;
using Yolo.OutputData;
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

	public void PredictPlotAndAssert(string modelFileName, ClassificationTestData data)
	{
		var classifications = Predict(modelFileName, data, out var predictor);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var actualClassification = predictor.Metadata.ClassesNames[classifications[0].ClassId];
		actualClassification.Should().Be(data.ExpectedClassification);
	}

	private IReadOnlyList<Classification> Predict(string modelFileName, ClassificationTestData data, out Predictor predictor)
	{
		predictor = TestPredictorCreator.CreatePredictor(modelFileName, _useGpu);
		using var image = TestImageLoader.LoadImage(data.ImageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var classifications = predictor.Predict(imageData.Span, Argb32InputProcessor.Instance, V8ClassificationProcessor.Instance);
		return classifications;
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}