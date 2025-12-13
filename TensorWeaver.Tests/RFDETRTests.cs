using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.InputProcessing;
using TensorWeaver.RFDETR;
using TensorWeaver.Tests.Helpers;

namespace TensorWeaver.Tests;

public sealed class RFDETRTests
{
	[Fact]
	public async Task ShouldDetect()
	{
		var modelPath = Path.Combine("Models", "rf-detr-nano.onnx");
		if (!File.Exists(modelPath))
		{
			using var client = new HttpClient();
			const string downloadModelPath = "https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx";
			var response = await client.GetAsync(downloadModelPath, TestContext.Current.CancellationToken);
			await using var targetStream = File.OpenWrite(modelPath);
			await response.Content.CopyToAsync(targetStream, TestContext.Current.CancellationToken);
		}
		var modelData = await File.ReadAllBytesAsync(modelPath, TestContext.Current.CancellationToken);
		var predictor = new Predictor(modelData, new SessionOptions());
		using var image = TestImageLoader.LoadImage("bus480.png");
		var imageData = TestImageLoader.ExtractImageData(image);
		predictor.SetInput(imageData.Span, new ResizingInputProcessor<Argb32>(ImageSharpInputProcessors.Argb32, new NearestNeighbourImageResizer()));
		predictor.Predict();
		var detections = predictor.GetOutput(new RFDETRDetectionProcessor());
		var plotted = DetectionsPlottingHelper.Plot(image, null, detections);
		ImageSaver.Save(plotted, "rf-detr-nano.onnx", "bus480.png", false, "ShouldDetect");
	}
}