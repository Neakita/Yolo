using System.Runtime.CompilerServices;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputData;
using Yolo.OutputProcessing;
using Yolo.Tests.Data;

namespace Yolo.Tests.Helpers;

public class PoseTestHelper
{
	public PoseTestHelper(ITestOutputHelper testOutputHelper, bool useGpu)
	{
		_testOutputHelper = testOutputHelper;
		_useGpu = useGpu;
	}

	public void PredictPlotAndAssert(DetectionTestData testData, [CallerMemberName] string testName = "")
	{
		using var image = TestImageLoader.LoadImage(testData.ImageName);
		var imageData = TestImageLoader.ExtractImageData(image);
		var poses = Predict(testData, imageData, out var predictor);
		var classifications = poses.Select(pose => pose.Classification).ToList();
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, predictor.Metadata, classifications);
		var plotted = PosePlottingHelper.Plot(image, predictor.Metadata, poses);
		ImageSaver.Save(plotted, testData.ModelName, testData.ImageName, _useGpu, $"{nameof(PoseTests)}.{testName}");
		DetectionAssertionHelper.AssertClassifications(predictor.Metadata, testData.ObjectsExpectations, classifications);
	}

	private IReadOnlyList<Pose> Predict(DetectionTestData testData, ReadOnlyMemory2D<Argb32> imageData, out Predictor predictor)
	{
		predictor = TestPredictorCreator.CreatePredictor(testData.ModelName, _useGpu);
		V8PoseProcessor outputProcessor = new(predictor)
		{
			MinimumConfidence = 0.5f
		};
		var poses = predictor.Predict(imageData.Span, Argb32InputProcessor.Instance, outputProcessor);
		return poses;
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}