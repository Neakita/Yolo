using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;

namespace Yolo.Tests;

public class PredictorTests
{
	[Theory]
	[InlineData("toaster.png", "toaster")]
	public void ClassificationTest(string imageFileName, string expectedClassName)
	{
		Predictor predictor = new(File.ReadAllBytes("Models/yolov8n-cls-uint8.onnx"), new SessionOptions());
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		var result = predictor.Predict(data.Span, new Rgb24InputProcessor(), new ClassificationOutputProcessor());
		Assert.Equal(predictor.Metadata.ClassesNames[result[0].ClassId], expectedClassName);
	}
}