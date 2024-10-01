using CommunityToolkit.Diagnostics;
using FluentAssertions;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Tests;

public class DetectionTests
{
	[Theory]
	[InlineData("bus.png", "person:4,bus:1")]
	public void DetectionTest(string imageFileName, string expectedResults)
	{
		Predictor predictor = new(File.ReadAllBytes("Models/yolov8n-uint8.onnx"), new SessionOptions());
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		predictor.Predict(data.Span, new Rgb24InputProcessor());
		var result = new V8DetectionOutputProcessor().Process(predictor.Output);
		var stringResults = result.GroupBy(x => x.Classification.ClassId)
			.OrderByDescending(x => x.Count())
			.Select(x => $"{predictor.Metadata.ClassesNames[x.Key]}:{x.Count()}");
		var stringResult = string.Join(',', stringResults);
		stringResult.Should().Be(expectedResults);
	}
}