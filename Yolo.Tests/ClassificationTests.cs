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
		var result = predictor.Predict(data.Span, new Rgb24InputProcessor(), new V8ClassificationOutputProcessor());
		Assert.Equal(predictor.Metadata.ClassesNames[result[0].ClassId], expectedClassName);
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
			var result = predictor.Predict(data.Span, new Rgb24InputProcessor(), new V8ClassificationOutputProcessor());
			var resultClassName = predictor.Metadata.ClassesNames[result[0].ClassId];
			Assert.Equal(resultClassName, expectedClassName);
		}
	}
}