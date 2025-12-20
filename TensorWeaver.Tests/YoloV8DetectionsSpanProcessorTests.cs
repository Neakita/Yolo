using FluentAssertions;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.OutputData;
using TensorWeaver.Tests.Inference;
using TensorWeaver.Tests.Inference.Detectors;
using TensorWeaver.Yolo;
using Xunit;

namespace TensorWeaver.Tests;

public class YoloV8DetectionsSpanProcessorTests
{
	[Fact]
	public async Task Span()
	{
		var predictor = new Predictor(await DetectorModels.YoloV8Nano.ReadDataAsync(TestContext.Current.CancellationToken), new SessionOptions());
		var imageInfo = new ImageInfo("bus.png");
		predictor.SetInput((await imageInfo.GetPixelsAsync<Argb32>()).Span, ImageSharpInputProcessors.Argb32.WithResizing());
		predictor.Predict();
		var metadata = YoloMetadata.Parse(predictor.Session);
		var processor = new YoloV8DetectionsSpanProcessor(metadata);
		var detections = new Detection[20];
		var detectionsCount = processor.Process(predictor.Output, detections);
		detectionsCount.Should().BeGreaterThan(0);
	}
}