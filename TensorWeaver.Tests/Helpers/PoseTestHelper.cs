using System.Runtime.CompilerServices;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.InputProcessing;
using TensorWeaver.OutputData;
using TensorWeaver.Tests.Data;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Helpers;

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
		var metadata = YoloMetadata.Parse(predictor.Session);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, metadata, classifications);
		var plotted = PosePlottingHelper.Plot(image, metadata, poses);
		ImageSaver.Save(plotted, testData.ModelName, testData.ImageName, _useGpu, $"{nameof(PoseTests)}.{testName}");
		DetectionAssertionHelper.AssertClassifications(metadata, testData.ObjectsExpectations, classifications);
	}

	private IReadOnlyList<Pose> Predict(DetectionTestData testData, ReadOnlyMemory2D<Argb32> imageData, out Predictor predictor)
	{
		predictor = TestPredictorCreator.CreatePredictor(testData.ModelName, _useGpu);
		V8PoseProcessor outputProcessor = new(predictor.Session)
		{
			MinimumConfidence = 0.5f
		};
		predictor.SetInput(imageData.Span, new ResizingInputProcessor<Argb32>(ImageSharpInputProcessors.Argb32, new NearestNeighbourImageResizer()));
		predictor.Predict();
		var poses = predictor.GetOutput(outputProcessor);
		return poses;
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;
}