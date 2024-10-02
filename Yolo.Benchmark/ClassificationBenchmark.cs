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
public class ClassificationBenchmark
{
	static ClassificationBenchmark()
	{
		SessionOptions options = new();
		options.AppendExecutionProvider_Tensorrt();
		Predictor = new Predictor(File.ReadAllBytes("Models/yolov8n-cls-uint8.onnx"), options);
		const string imageFilePath = "Images/toaster.png";
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		ImageData = data.ToArray();
	}

	[Benchmark]
	public Classification Predict()
	{
		var output = Predictor.Predict(ImageData, InputProcessor);
		return Processor.Process(output).First();
	}

	private static readonly Predictor Predictor;
	private static readonly Rgb24[] ImageData;
	private static readonly Rgb24InputProcessor InputProcessor = new();
	private static readonly V8ClassificationProcessor Processor = new();
}