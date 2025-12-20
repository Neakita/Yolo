using BenchmarkDotNet.Attributes;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.Benchmark.Models;
using TensorWeaver.ImageSharp;
using TensorWeaver.OutputData;
using TensorWeaver.Yolo;

namespace TensorWeaver.Benchmark;

public class ClassificationProcessorsBenchmark
{
	static ClassificationProcessorsBenchmark()
	{
		Predictor = new Predictor(ClassificationModels.All.First().ReadData(), new SessionOptions());
		Predictor.SetInput(Images.Bus.GetPixels<Argb32>().Span, ImageSharpInputProcessors.Argb32.WithResizing());
	}

	[Benchmark]
	public Classification[] Array()
	{
		return ArrayProcessor.Process(Predictor.Output);
	}

	[Benchmark]
	public Classification Single()
	{
		return SingleProcessor.Process(Predictor.Output);
	}

	[Benchmark]
	public int Span()
	{
		Span<Classification> classifications = stackalloc Classification[5];
		SpanProcessor.Process(Predictor.Output, classifications);
		return classifications.Length;
	}

	private static readonly Predictor Predictor;
	private static readonly YoloClassificationsProcessor ArrayProcessor = new();
	private static readonly YoloClassificationProcessor SingleProcessor = new();
	private static readonly YoloClassificationsSpanProcessor SpanProcessor = new();
}