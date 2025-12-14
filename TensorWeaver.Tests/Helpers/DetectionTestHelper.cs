using System.Runtime.CompilerServices;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.InputProcessing;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;
using TensorWeaver.Tests.Data;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Helpers;

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
		var metadata = YoloMetadata.Parse(predictor.Session);
		DetectionsOutputHelper.WriteClassifications(_testOutputHelper, metadata, classifications);
		var plotted = DetectionsPlottingHelper.Plot(image, metadata.ClassesNames, detections);
		ImageSaver.Save(plotted, testData.ModelName, testData.ImageName, _useGpu, testName);
		DetectionAssertionHelper.AssertClassifications(metadata, testData.ObjectsExpectations, classifications);
	}

	private readonly ITestOutputHelper _testOutputHelper;
	private readonly bool _useGpu;

	private IReadOnlyList<Detection> Predict(DetectionTestData testData, ReadOnlyMemory2D<Argb32> imageData, out Predictor predictor)
	{
		predictor = TestPredictorCreator.CreatePredictor(testData.ModelName, _useGpu);
		var metadata = YoloMetadata.Parse(predictor.Session);
		OutputProcessor<IReadOnlyList<Detection>> outputProcessor = metadata.Version switch
		{
			8 => new YoloV8DetectionsProcessor(metadata),
			10 => new YoloV10DetectionsProcessor(metadata),
			_ => throw new ArgumentOutOfRangeException()
		};
		predictor.SetInput(imageData.Span, new ResizingInputProcessor<Argb32>(ImageSharpInputProcessors.Argb32, new NearestNeighbourImageResizer()));
		predictor.Predict();
		var detections = predictor.GetOutput(outputProcessor);
		return detections;
	}
}