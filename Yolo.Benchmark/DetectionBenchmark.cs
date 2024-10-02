using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Benchmark;

[MemoryDiagnoser]
[EventPipeProfiler(EventPipeProfile.CpuSampling)]
public class DetectionBenchmark
{
	[Params("yolov8n-uint8.onnx", "yolov8n320fp16.onnx", "yolov8n320fp32.onnx", "yolov8n320int8.onnx")]
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
		Predictor = new Predictor(File.ReadAllBytes(Path.Combine("Models", ModelName)), options);
		var imageFileName = Predictor.Metadata.ImageSize.Width == 320 ? "bus320.png" : "bus640.png";
		var image = Image.Load<Rgb24>(Path.Combine("Images", imageFileName));
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		ImageData = data.ToArray();
		OutputProcessor = new V8DetectionProcessor(Predictor.Metadata);
	}

	[Benchmark]
	public IReadOnlyList<Detection> Predict()
	{
		return Predictor.Predict(ImageData, InputProcessor, OutputProcessor);
	}

	private Predictor Predictor = null!;
	private Rgb24[] ImageData = null!;
	private static readonly Rgb24InputProcessor InputProcessor = new();
	private V8DetectionProcessor OutputProcessor = null!;
}