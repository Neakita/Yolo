using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;
using TensorWeaver.RFDETR;
using TensorWeaver.Yolo;

namespace TensorWeaver.Benchmark;

public class RFDETRBenchmark
{
	[Params("Cpu"/*, "Cuda", "TensorRT"*/)] public string ExecutionProvider { get; set; } = null!;

	[GlobalSetup]
	public void Setup()
	{
		SessionOptions options = new();
		switch (ExecutionProvider)
		{
			case "Cpu":
				break;
			case "Cuda":
				options.AppendExecutionProvider_CUDA();
				break;
			case "TensorRT":
				options.AppendExecutionProvider_Tensorrt();
				break;
			default:
				throw new ArgumentException($"Unknown ExecutionProvider: {ExecutionProvider}");
		}
		var modelPath = Path.Combine("Models", "rf-detr-nano.onnx");
		if (!File.Exists(modelPath))
		{
			using var client = new HttpClient();
			const string downloadModelPath = "https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx";
			var response = client.GetAsync(downloadModelPath).GetAwaiter().GetResult();
			using var targetStream = File.OpenWrite(modelPath);
			response.Content.CopyTo(targetStream, null, CancellationToken.None);
		}
		_predictor = new Predictor(File.ReadAllBytes(modelPath), options);
		var metadata = YoloMetadata.Parse(_predictor.Session);
		var size = metadata.ImageSize.X;
		var imageFileName = $"bus{size}.png";
		var image = Image.Load<Argb32>(Path.Combine("Images", imageFileName));
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		_imageData = data.ToArray();
		_imageSize = new Vector2D<int>(image.Width, image.Height);
	}

	[GlobalCleanup]
	public void CleanUp()
	{
		_predictor.Dispose();
	}

	[Benchmark]
	public IReadOnlyList<Detection> Predict()
	{
		_predictor.SetInput(new ReadOnlySpan2D<Argb32>(_imageData, _imageSize.Y, _imageSize.X), ImageSharpInputProcessors.Argb32);
		_predictor.Predict();
		var result = _predictor.GetOutput(_outputProcessor);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private Predictor _predictor = null!;
	private Argb32[] _imageData = null!;
	private Vector2D<int> _imageSize;
	private readonly OutputProcessor<List<Detection>> _outputProcessor = new RFDETRDetectionProcessor();
}