using System.Runtime.CompilerServices;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

public class PoseTestHelper
{
	// 2 persons are clearly visible and 2 more are only partially.
	// the model doesn't know about bus and stop sign, as detection models, so it is excluded
	// the sign is barely noticeable
	public const string BusExpectedPrediction = "person:2-4";

	public PoseTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(string modelFileName, string imageFileName, string expectedDetections, [CallerMemberName] string testName = "")
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, _useGpu);
		V8PoseProcessor outputProcessor = new(predictor)
		{
			MinimumConfidence = 0.5f
		};
		var image = TestImageLoader.LoadImage(imageFileName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var poses = predictor.Predict(imageData.Span2D, Rgb24InputProcessor.Instance, outputProcessor);
		var classifications = poses.Select(pose => pose.Classification).ToList();
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var plotted = PosePlottingHelper.Plot(image, predictor.Metadata, poses);
		ImageSaver.Save(plotted, modelFileName, imageFileName, _useGpu, $"{nameof(PoseTests)}.{testName}");
		DetectionAssertionHelper.AssertClassifications(predictor.Metadata, expectedDetections, classifications);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}