using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Tests;

public sealed class ClassificationTests
{
	[Theory]
	[InlineData("toaster.png", "toaster")]
	public void ShouldClassify(string imageFileName, string expectedClassName)
	{
		Predictor predictor = new(File.ReadAllBytes("Models/yolov8n-cls-uint8.onnx"), new SessionOptions());
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		predictor.Predict(data.Span, new Rgb24InputProcessor());
		var result = new V8ClassificationProcessor().Process(predictor.Output).First();
		Assert.Equal(predictor.Metadata.ClassesNames[result.ClassId], expectedClassName);
	}

	[Theory]
	[InlineData("toaster.png;pizza.png", "toaster;pizza")]
	public void ShouldClassifySequentially(string imageFilesNames, string expectedClassesNames)
	{
		Predictor predictor = new(File.ReadAllBytes("Models/yolov8n-cls-uint8.onnx"), new SessionOptions());
		var images = imageFilesNames
			.Split(';')
			.Select(fileName => Path.Combine("Images", fileName))
			.Select(Image.Load<Rgb24>)
			.ToList();
		var expectations = expectedClassesNames.Split(';');
		foreach (var (image, expectedClassName) in images.Zip(expectations))
		{
			Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
			predictor.Predict(data.Span, new Rgb24InputProcessor());
			var result = new V8ClassificationProcessor().Process(predictor.Output).First();
			var resultClassName = predictor.Metadata.ClassesNames[result.ClassId];
			Assert.Equal(resultClassName, expectedClassName);
		}
	}

	[Theory]
	[InlineData("toaster.png")]
	public void ShouldNotEnumerateProcessedResultsAfterAnotherPrediction(string imageFileName)
	{
		Predictor predictor = new(File.ReadAllBytes("Models/yolov8n-cls-uint8.onnx"), new SessionOptions());
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		predictor.Predict(data.Span, new Rgb24InputProcessor());
		V8ClassificationProcessor processor = new();
		processor.MinimumConfidence = 0;
		var enumerableResult = processor.Process(predictor.Output);
		Assert.ThrowsAny<Exception>(() =>
		{
			foreach (var dummy in enumerableResult)
				predictor.Predict(data.Span, new Rgb24InputProcessor());
		});
	}
}