using Microsoft.ML.OnnxRuntime;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;
using TensorWeaver.Tests.Inference.Detectors;

namespace TensorWeaver.Tests.Inference;

public sealed class TestCase
{
	public static IEnumerable<TestCase> Create(ModelInfo model, ImageInfo image, OutputProcessor<IReadOnlyCollection<Detection>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		yield return Create(model, image, InferenceTestSession.CPU, outputProcessor, expectations);
		yield return Create(model, image, InferenceTestSession.Cuda, outputProcessor, expectations);
	}

	private static TestCase Create(ModelInfo model, ImageInfo image, InferenceTestSession session, OutputProcessor<IReadOnlyCollection<Detection>> outputProcessor, IReadOnlyCollection<DetectedObjectExpectation> expectations)
	{
		var modelName = Path.GetFileNameWithoutExtension(model.FileName);
		var imageName = Path.GetFileNameWithoutExtension(image.FileName);
		var imageExtension = Path.GetExtension(image.FileName);
		var plottedFileName = $"{imageName}-{modelName}-{session.Designation}.{imageExtension}";
		var plotter = new DetectionsPlotter(image, model.ClassesNames, plottedFileName);
		var asserter = new DetectionsAsserter(expectations, model.ClassesNames);
		var resultHandler = new CompositeResultHandler<IReadOnlyCollection<Detection>>(plotter, asserter);
		var outputHandler = new ProcessingPredictorOutputHandler<IReadOnlyCollection<Detection>>(outputProcessor, resultHandler);
		return new TestCase
		{
			Model = model,
			SessionOptionsFactory = session.Factory,
			ImageInfo = image,
			OutputHandler = outputHandler
		};
	}

	public required ModelInfo Model { get; init; }
	public required Func<SessionOptions> SessionOptionsFactory { get; init; }
	public required ImageInfo ImageInfo { get; init; }
	public required PredictorOutputHandler OutputHandler { get; init; }
}