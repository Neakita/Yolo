using System.Runtime.CompilerServices;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputData;
using Yolo.OutputProcessing;
using Yolo.Tests.Data;

namespace Yolo.Tests.Helpers;

internal class DetectionTestHelper
{
	public DetectionTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(DetectionTestData testData, [CallerMemberName] string testName = "")
	{
		using var image = TestImageLoader.LoadImage(testData.ImageName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var detections = Predict(testData, imageData, out var predictor);
		var classifications = detections.Select(detection => detection.Classification).ToList();
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var plotted = DetectionsPlottingHelper.Plot(image, predictor.Metadata, detections);
		ImageSaver.Save(plotted, testData.ModelName, testData.ImageName, _useGpu, testName);
		DetectionAssertionHelper.AssertClassifications(predictor.Metadata, testData.ObjectsExpectations, classifications);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;

	private IReadOnlyList<Detection> Predict(DetectionTestData testData, ReadOnlyMemory2D<Argb32> imageData, out Predictor predictor)
	{
		predictor = TestPredictorCreator.CreatePredictor(testData.ModelName, _useGpu);
		OutputProcessor<Detection> outputProcessor = predictor.Metadata.Version switch
		{
			8 => new V8DetectionProcessor(predictor.Metadata),
			10 => new V10DetectionProcessor(predictor.Metadata),
			_ => throw new ArgumentOutOfRangeException()
		};
		outputProcessor.MinimumConfidence = 0.5f;
		var detections = predictor.Predict(imageData.Span, Argb32InputProcessor.Instance, outputProcessor);
		return detections;
	}
}