using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.Benchmark;

public class CompunetClassificationBenchmark
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

		YoloPredictorOptions yoloPredictorOptions = new()
		{
			UseCuda = false,
			SessionOptions = options
		};
		_predictor = new YoloPredictor(Path.Combine("Models", ModelName), yoloPredictorOptions);
		const string imageFileName = "pizza224.png";
		var image = Image.Load<Rgb24>(Path.Combine("Images", imageFileName));
		_image = image;
	}

	[GlobalCleanup]
	public void CleanUp()
	{
		_predictor.Dispose();
		_image.Dispose();
	}


	[Benchmark]
	public YoloResult<Compunet.YoloSharp.Data.Classification> Predict()
	{
		var result = _predictor.Classify(_image);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private YoloPredictor _predictor = null!;
	private Image<Rgb24> _image = null!;
}