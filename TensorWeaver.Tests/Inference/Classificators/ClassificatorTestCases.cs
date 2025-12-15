using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Inference.Classificators;

public static class ClassificatorTestCases
{
	public static IEnumerable<TestCase> All =>
	[
		..YoloV8Nano,
		..YoloV8NanoUInt8
	];

	private static IEnumerable<TestCase> YoloV8Nano =>
		Create(ClassificatorModels.YoloV8Nano, 
			PizzaImageInfo, 
			new YoloV8ClassificationsProcessor(), 
			"pizza");

	private static IEnumerable<TestCase> YoloV8NanoUInt8 =>
		Create(ClassificatorModels.YoloV8NanoUInt8, 
			PizzaImageInfo, 
			new YoloV8ClassificationsProcessor(), 
			"pizza");
	
	private static readonly ImageInfo PizzaImageInfo = new("pizza.png");

	private static IEnumerable<TestCase> Create(ModelInfo model, ImageInfo image, OutputProcessor<IReadOnlyCollection<Classification>> outputProcessor, string expectedClassName)
	{
		yield return Create(model, image, InferenceTestSession.CPU, outputProcessor, expectedClassName);
		yield return Create(model, image, InferenceTestSession.Cuda, outputProcessor, expectedClassName);
	}

	private static TestCase Create(ModelInfo model, ImageInfo image, InferenceTestSession session, OutputProcessor<IReadOnlyCollection<Classification>> outputProcessor, string expectedClassName)
	{
		var modelName = Path.GetFileNameWithoutExtension(model.FileName);
		var imageName = Path.GetFileNameWithoutExtension(image.FileName);
		var imageExtension = Path.GetExtension(image.FileName);
		var plottedFileName = $"{imageName}-{modelName}-{session.Designation}{imageExtension}";
		var plotter = new ClassificationsPlotter(image, model.ClassesNames, plottedFileName);
		var asserter = new ClassificationAsserter(model.ClassesNames, expectedClassName);
		var resultHandler = new CompositeResultHandler<IReadOnlyCollection<Classification>>(plotter, asserter);
		var outputHandler = new ProcessingPredictorOutputHandler<IReadOnlyCollection<Classification>>(outputProcessor, resultHandler);
		return new TestCase
		{
			Model = model,
			Session = session,
			Image = image,
			OutputHandler = outputHandler
		};
	}
}