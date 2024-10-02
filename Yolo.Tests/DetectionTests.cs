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
	[InlineData("yolov8n-uint8.onnx", "bus640.png", "person:4,bus:1")]
	public void DetectionTest(string modelName, string imageFileName, string expectedResults)
	{
		SessionOptions options = new();
		options.AppendExecutionProvider_CUDA();
		options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
		Predictor predictor = new(File.ReadAllBytes(Path.Combine("Models", modelName)), options);
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		var output = predictor.Predict(data.Span, new Rgb24InputProcessor());
		var result = new V8DetectionProcessor(predictor.Metadata).Process(output);
		var stringResults = result.GroupBy(x => x.Classification.ClassId)
			.OrderByDescending(x => x.Count())
			.Select(x => $"{predictor.Metadata.ClassesNames[x.Key]}:{x.Count()}");
		var stringResult = string.Join(',', stringResults);
		stringResult.Should().Be(expectedResults);
	}
}