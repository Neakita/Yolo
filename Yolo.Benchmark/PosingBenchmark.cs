using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Benchmark;

public class PosingBenchmark
{
	[Params(
		"yolov8n-pose-uint8.onnx",
		"yolov8n-pose160fp32.onnx",
		"yolov8n-pose224fp32.onnx",
		"yolov8n-pose320fp32.onnx",
		"yolov8n-pose480fp32.onnx",
		"yolov8n-pose640fp32.onnx",
		"yolov8n-pose800fp32.onnx",
		"yolo11n-pose160fp32.onnx",
		"yolo11n-pose224fp32.onnx",
		"yolo11n-pose320fp32.onnx",
		"yolo11n-pose480fp32.onnx",
		"yolo11n-pose640fp32.onnx",
		"yolo11n-pose800fp32.onnx")]
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
		var image = Image.Load<Argb32>(Path.Combine("Images", imageFileName));
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		_imageData = data.ToArray();
		_outputProcessor = new V8PoseProcessor(_predictor);
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
	public IReadOnlyList<Pose> Predict()
	{
		var result = _predictor.Predict(new ReadOnlySpan2D<Argb32>(_imageSize, _imageData), Argb32InputProcessor.Instance, _outputProcessor);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private Predictor _predictor = null!;
	private Argb32[] _imageData = null!;
	private Vector2D<int> _imageSize;
	private OutputProcessor<Pose> _outputProcessor = null!;
}