using System.Runtime.CompilerServices;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;
using Yolo.Tests.Data;

namespace Yolo.Tests.Helpers;

public class PoseTestHelper
{
	// 2 persons are clearly visible and 2 more are only partially.
	// the model doesn't know about bus and stop sign, so it is excluded
	public const string BusExpectedPrediction = "person:2-4";

	public PoseTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(DetectionTestData testData, [CallerMemberName] string testName = "")
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(testData.ModelName, _useGpu);
		V8PoseProcessor outputProcessor = new(predictor)
		{
			MinimumConfidence = 0.5f
		};
		var image = TestImageLoader.LoadImage(testData.ImageName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var poses = predictor.Predict(imageData.Span, Argb32InputProcessor.Instance, outputProcessor);
		var classifications = poses.Select(pose => pose.Classification).ToList();
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var plotted = PosePlottingHelper.Plot(image, predictor.Metadata, poses);
		ImageSaver.Save(plotted, testData.ModelName, testData.ImageName, _useGpu, $"{nameof(PoseTests)}.{testName}");
		DetectionAssertionHelper.AssertClassifications(predictor.Metadata, testData.ObjectsExpectations, classifications);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}