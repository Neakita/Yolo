using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;
using Yolo.OutputData;
using Yolo.OutputProcessing;

namespace Yolo.Benchmark;

public class ClassificationBenchmark
{
	[Params("yolov8n224fp32cls.onnx", "yolo11n224fp32cls.onnx")]
	public string ModelName { get; set; } = null!;

	[Params("Cpu", "Cuda", "TensorRT")] public string ExecutionProvider { get; set; } = null!;

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

		_predictor = new Predictor(File.ReadAllBytes(Path.Combine("Models", ModelName)), options);
		const string imageFileName = "pizza224.png";
		var image = Image.Load<Argb32>(Path.Combine("Images", imageFileName));
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		_imageData = data.ToArray();
		_outputProcessor = new V8ClassificationProcessor();
		_imageSize = new Vector2D<int>(image.Width, image.Height);
	}

	[GlobalCleanup]
	public void CleanUp()
	{
		_predictor.Dispose();
		if (_outputProcessor is IDisposable disposable)
			disposable.Dispose();
	}

	[Benchmark]
	public IReadOnlyList<Classification> Predict()
	{
		var result = _predictor.Predict(new ReadOnlySpan2D<Argb32>(_imageData, _imageSize.Y, _imageSize.X), Argb32InputProcessor.Instance, _outputProcessor);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private Predictor _predictor = null!;
	private Argb32[] _imageData = null!;
	private Vector2D<int> _imageSize;
	private OutputProcessor<Classification> _outputProcessor = null!;
}