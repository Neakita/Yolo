using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.Benchmark;

public class CompunetDetectionBenchmark
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

		YoloPredictorOptions yoloPredictorOptions = new()
		{
			UseCuda = false,
			SessionOptions = options
		};
		_predictor = new YoloPredictor(File.ReadAllBytes(Path.Combine("Models", ModelName)), yoloPredictorOptions);
		var size = _predictor.Metadata.ImageSize.Width;
		var imageFileName = $"bus{size}.png";
		_image = Image.Load<Rgb24>(Path.Combine("Images", imageFileName));
	}

	[GlobalCleanup]
	public void CleanUp()
	{
		_predictor.Dispose();
		_image.Dispose();
	}

	[Benchmark]
	public YoloResult<Compunet.YoloSharp.Data.Detection> Predict()
	{
		var result = _predictor.Detect(_image);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private YoloPredictor _predictor = null!;
	private Image<Rgb24> _image = null!;
}