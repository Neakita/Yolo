using System.Collections.Immutable;
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
		Predictor = new Predictor(File.ReadAllBytes("Models/yolov8n-uint8.onnx"), new SessionOptions());
		const string imageFilePath = "Images/bus.png";
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		ImageData = data.ToArray();
	}

	[Benchmark]
	public ImmutableArray<Detection> Predict()
	{
		return Predictor.Predict(ImageData, InputProcessor, OutputProcessor);
	}

	private static readonly Predictor Predictor;
	private static readonly Rgb24[] ImageData;
	private static readonly Rgb24InputProcessor InputProcessor = new();
	private static readonly V8DetectionOutputProcessor OutputProcessor = new();
}