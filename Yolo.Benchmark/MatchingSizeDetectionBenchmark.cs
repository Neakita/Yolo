using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Benchmark;

[MemoryDiagnoser]
public class MatchingSizeDetectionBenchmark
{
	[Params(
		"yolov8n-uint8.onnx",
		"yolov8n160fp32.onnx",
		"yolov8n224fp32.onnx",
		"yolov8n320fp32.onnx",
		"yolov8n480fp32.onnx",
		"yolov8n640fp32.onnx",
		"yolov8n800fp32.onnx",
		"yolov10n160fp32.onnx",
		"yolov10n224fp32.onnx",
		"yolov10n320fp32.onnx",
		"yolov10n480fp32.onnx",
		"yolov10n640fp32.onnx",
		"yolov10n800fp32.onnx",
		"yolo11n160fp32.onnx",
		"yolo11n224fp32.onnx",
		"yolo11n320fp32.onnx",
		"yolo11n480fp32.onnx",
		"yolo11n640fp32.onnx",
		"yolo11n800fp32.onnx")]
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
		var size = _predictor.Metadata.ImageSize.X;
		var imageFileName = $"bus{size}.png";
		var image = Image.Load<Rgb24>(Path.Combine("Images", imageFileName));
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		_imageData = data.ToArray();
		_outputProcessor = _predictor.Metadata.ModelVersion switch
		{
			8 => new V8DetectionProcessor(_predictor.Metadata),
			10 => new V10DetectionProcessor(_predictor.Metadata),
			_ => throw new ArgumentOutOfRangeException()
		};
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
	public IReadOnlyList<Detection> Predict()
	{
		var result = _predictor.Predict(new ReadOnlySpan2D<Rgb24>(_imageSize, _imageData), Rgb24InputProcessor.Instance, _outputProcessor);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private Predictor _predictor = null!;
	private Rgb24[] _imageData = null!;
	private Vector2D<int> _imageSize;
	private OutputProcessor<Detection> _outputProcessor = null!;
}