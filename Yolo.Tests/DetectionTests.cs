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
	[InlineData("yolov8n-uint8.onnx", "bus640.png", "person:4,bus:1", true)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", "person:3,bus:1", false)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", "person:3,bus:1", true)]
	[InlineData("yolov8n320int8.onnx", "bus320.png", "person:3,bus:1", false)]
	[InlineData("yolov8n320int8.onnx", "bus320.png", "person:3,bus:1", true)]
	[InlineData("yolov8n320fp16.onnx", "bus320.png", "person:3,bus:1", false)]
	[InlineData("yolov8n320fp16.onnx", "bus320.png", "person:3,bus:1", true)]
	public void DetectionTest(string modelName, string imageFileName, string expectedResults, bool gpu)
	{
		SessionOptions options = new();
		if (gpu)
			options.AppendExecutionProvider_CUDA();
		Predictor predictor = new(File.ReadAllBytes(Path.Combine("Models", modelName)), options);
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		var result = predictor.Predict(data.Span, new Rgb24InputProcessor(), new V8DetectionProcessor(predictor.Metadata));
		var stringResults = result.GroupBy(x => x.Classification.ClassId)
			.OrderByDescending(x => x.Count())
			.Select(x => $"{predictor.Metadata.ClassesNames[x.Key]}:{x.Count()}");
		var stringResult = string.Join(',', stringResults);
		stringResult.Should().Be(expectedResults);
	}
}