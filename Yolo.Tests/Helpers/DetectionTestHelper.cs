using System.Runtime.CompilerServices;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Tests.Helpers;

internal class DetectionTestHelper
{
	// 2 persons are clearly visible and 2 more are only partially.
	// bus is huge object on the image and locates in the center
	// the sign is barely noticeable
	public const string BusExpectedPrediction = "person:2-4,bus:1,stop sign:0-1";

	public DetectionTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(string modelFileName, string imageFileName, string expectedDetections, [CallerMemberName] string testName = "")
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, _useGpu);
		OutputProcessor<Detection> outputProcessor = predictor.Metadata.ModelVersion switch
		{
			8 => new V8DetectionProcessor(predictor.Metadata),
			10 => new V10DetectionProcessor(predictor.Metadata),
			_ => throw new ArgumentOutOfRangeException()
		};
		outputProcessor.MinimumConfidence = 0.5f;
		var image = TestImageLoader.LoadImage(imageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var detections = predictor.Predict(imageData.Span, Argb32InputProcessor.Instance, outputProcessor);
		var classifications = detections.Select(detection => detection.Classification).ToList();
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var plotted = DetectionsPlottingHelper.Plot(image, predictor.Metadata, detections);
		ImageSaver.Save(plotted, modelFileName, imageFileName, _useGpu, testName);
		DetectionAssertionHelper.AssertClassifications(predictor.Metadata, expectedDetections, classifications);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}