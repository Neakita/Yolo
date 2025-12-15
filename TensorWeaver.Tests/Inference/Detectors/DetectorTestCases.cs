using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;
using TensorWeaver.RFDETR;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Inference.Detectors;

internal static class DetectorTestCases
{
	public static IEnumerable<TestCase> All =>
	[
		..RFDETRNano,
		..YoloV8Nano,
		..YoloV8NanoUInt8,
		..YoloV10Nano,
		..YoloV11Nano
	];

	private static IEnumerable<TestCase> RFDETRNano =>
		Create(DetectorModels.RFDETRNano,
			BusImageInfo,
			new RFDETRDetectionProcessor(),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV8Nano =>
		Create(DetectorModels.YoloV8Nano, 
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)DetectorModels.YoloV8Nano.ClassesNames.Count,
				new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV8NanoUInt8 =>
		Create(DetectorModels.YoloV8NanoUInt8,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)DetectorModels.YoloV8NanoUInt8.ClassesNames.Count,
				new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV10Nano =>
		Create(DetectorModels.YoloV10Nano, 
			BusImageInfo,
			new YoloV10DetectionsProcessor(new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV11Nano =>
		Create(DetectorModels.YoloV11Nano,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)DetectorModels.YoloV8NanoUInt8.ClassesNames.Count,
				new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static readonly IReadOnlyCollection<DetectedObjectExpectation> BusImageObjectExpectations =
	[
		new("person", 2, 4), // 2 persons are clearly visible and 2 more are only partially
		new("bus", 1), // bus is huge object on the image and locates in the center
		new("stop sign", 0, 1) // the sign is barely noticeable
	];

	private static readonly ImageInfo BusImageInfo = new("bus.png");

	private static IEnumerable<TestCase> Create(ModelInfo model, ImageInfo image, OutputProcessor<IReadOnlyCollection<Detection>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		yield return Create(model, image, InferenceTestSession.CPU, outputProcessor, expectations);
		yield return Create(model, image, InferenceTestSession.Cuda, outputProcessor, expectations);
	}

	private static TestCase Create(ModelInfo model, ImageInfo image, InferenceTestSession session, OutputProcessor<IReadOnlyCollection<Detection>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		var modelName = Path.GetFileNameWithoutExtension(model.FileName);
		var imageName = Path.GetFileNameWithoutExtension(image.FileName);
		var imageExtension = Path.GetExtension(image.FileName);
		var plottedFileName = $"{imageName}-{modelName}-{session.Designation}{imageExtension}";
		var plotter = new DetectionsPlotter(image, model.ClassesNames, plottedFileName);
		var asserter = new DetectionsAsserter(expectations, model.ClassesNames);
		var resultHandler = new CompositeResultHandler<IReadOnlyCollection<Detection>>(plotter, asserter);
		var outputHandler = new ProcessingPredictorOutputHandler<IReadOnlyCollection<Detection>>(outputProcessor, resultHandler);
		return new TestCase
		{
			Model = model,
			Session = session,
			Image = image,
			OutputHandler = outputHandler
		};
	}
}