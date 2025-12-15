using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;
using TensorWeaver.Tests.Inference.Detectors;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Inference.Posers;

internal static class PoseTestCases
{
	public static IEnumerable<TestCase> All =>
	[
		..YoloV8Nano,
		..YoloV8NanoUInt8
	];

	private static IEnumerable<TestCase> YoloV8Nano =>
		Create(PoserModels.YoloV8Nano,
			BusImageInfo,
			new YoloV8PosesProcessor(17, 3, 1, new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV8NanoUInt8 =>
		Create(PoserModels.YoloV8NanoUInt8,
			BusImageInfo,
			new YoloV8PosesProcessor(17, 3, 1, new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static readonly IReadOnlyCollection<DetectedObjectExpectation> BusImageObjectExpectations =
	[
		new("person", 2, 4) // 2 persons are clearly visible and 2 more are only partially
	];

	private static readonly ImageInfo BusImageInfo = new("bus.png");

	private static IEnumerable<TestCase> Create(ModelInfo model, ImageInfo image, OutputProcessor<IReadOnlyCollection<Pose>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		yield return Create(model, image, InferenceTestSession.CPU, outputProcessor, expectations);
		yield return Create(model, image, InferenceTestSession.Cuda, outputProcessor, expectations);
	}

	private static TestCase Create(ModelInfo model, ImageInfo image, InferenceTestSession session, OutputProcessor<IReadOnlyCollection<Pose>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		var modelName = Path.GetFileNameWithoutExtension(model.FileName);
		var imageName = Path.GetFileNameWithoutExtension(image.FileName);
		var imageExtension = Path.GetExtension(image.FileName);
		var plottedFileName = $"{imageName}-{modelName}-{session.Designation}{imageExtension}";
		var plotter = new PosesPlotter(image, model.ClassesNames, plottedFileName);
		var detectionsAsserter = new DetectionsAsserter(expectations, model.ClassesNames);
		var asserter = new PosesAsDetectionsHandlerAdapter(detectionsAsserter);
		var resultHandler = new CompositeResultHandler<IReadOnlyCollection<Pose>>(plotter, asserter);
		var outputHandler = new ProcessingPredictorOutputHandler<IReadOnlyCollection<Pose>>(outputProcessor, resultHandler);
		return new TestCase
		{
			Model = model,
			SessionOptionsFactory = session.Factory,
			ImageInfo = image,
			OutputHandler = outputHandler
		};
	}
}