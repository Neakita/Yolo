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
			DetectionsExpectations.BusCOCOExpectations);

	private static IEnumerable<TestCase> YoloV8Nano =>
		Create(DetectorModels.YoloV8Nano, 
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)DetectorModels.YoloV8Nano.ClassesNames.Count,
				new Vector2D<int>(800, 800)),
			DetectionsExpectations.BusCOCOExpectations);

	private static IEnumerable<TestCase> YoloV8NanoUInt8 =>
		Create(DetectorModels.YoloV8NanoUInt8,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)DetectorModels.YoloV8NanoUInt8.ClassesNames.Count,
				new Vector2D<int>(800, 800)),
			DetectionsExpectations.BusCOCOExpectations);

	private static IEnumerable<TestCase> YoloV10Nano =>
		Create(DetectorModels.YoloV10Nano, 
			BusImageInfo,
			new YoloV10DetectionsProcessor(new Vector2D<int>(800, 800)),
			DetectionsExpectations.BusCOCOExpectations);

	private static IEnumerable<TestCase> YoloV11Nano =>
		Create(DetectorModels.YoloV11Nano,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)DetectorModels.YoloV8NanoUInt8.ClassesNames.Count,
				new Vector2D<int>(800, 800)),
			DetectionsExpectations.BusCOCOExpectations);

	private static readonly ImageInfo BusImageInfo = new("bus.png");

	private static IEnumerable<TestCase> Create(ModelInfo model, ImageInfo image, OutputProcessor<IReadOnlyCollection<Detection>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		yield return Create(model, image, TestExecutionProvider.CPU, outputProcessor, expectations);
		yield return Create(model, image, TestExecutionProvider.Cuda, outputProcessor, expectations);
	}

	private static TestCase Create(ModelInfo model, ImageInfo image, TestExecutionProvider executionProvider, OutputProcessor<IReadOnlyCollection<Detection>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		var modelName = Path.GetFileNameWithoutExtension(model.FileName);
		var imageName = Path.GetFileNameWithoutExtension(image.FileName);
		var imageExtension = Path.GetExtension(image.FileName);
		var plottedFileName = $"{imageName}-{modelName}-{executionProvider.Designation}{imageExtension}";
		var plotter = new DetectionsPlotter(image, model.ClassesNames, plottedFileName);
		var asserter = new DetectionsAsserter(expectations, model.ClassesNames);
		var resultHandler = new CompositeResultHandler<IReadOnlyCollection<Detection>>(plotter, asserter);
		var outputHandler = new ProcessingPredictorOutputHandler<IReadOnlyCollection<Detection>>(outputProcessor, resultHandler);
		return new TestCase
		{
			Model = model,
			ExecutionProvider = executionProvider,
			Image = image,
			OutputHandler = outputHandler
		};
	}
}