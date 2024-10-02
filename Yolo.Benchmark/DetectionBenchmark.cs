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
	static DetectionBenchmark()
	{
		SessionOptions options = new();
		options.AppendExecutionProvider_CUDA();
		Predictor = new Predictor(File.ReadAllBytes("Models/yolov8n640fp32.onnx"), options);
		const string imageFilePath = "Images/bus320.png";
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		ImageData = data.ToArray();
		Processor = new V8DetectionProcessor(Predictor.Metadata);
	}

	[Benchmark]
	public Detection Predict()
	{
		var output = Predictor.Predict(ImageData, InputProcessor);
		return Processor.Process(output).First();
	}

	private static readonly Predictor Predictor;
	private static readonly Rgb24[] ImageData;
	private static readonly Rgb24InputProcessor InputProcessor = new();
	private static readonly V8DetectionProcessor Processor;
}