using FluentAssertions;
using TensorWeaver.ImageSharp;
using TensorWeaver.OutputData;
using TensorWeaver.Tests.Data;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Helpers;

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
		var metadata = YoloMetadata.Parse(predictor.Session);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, metadata, classifications);
		var actualClassification = metadata.ClassesNames[classifications[0].ClassId];
		actualClassification.Should().Be(data.ExpectedClassification);
	}

	private IReadOnlyList<Classification> Predict(string modelFileName, ClassificationTestData data, out Predictor predictor)
	{
		predictor = TestPredictorCreator.CreatePredictor(modelFileName, _useGpu);
		using var image = TestImageLoader.LoadImage(data.ImageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		predictor.SetInput(imageData.Span, ImageSharpInputProcessors.Argb32);
		predictor.Predict();
		var classifications = predictor.GetOutput(new YoloV8ClassificationsProcessor());
		return classifications;
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}