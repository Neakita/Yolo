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

	[Benchmark]
	public int SelectingSpan()
	{
		Span<Classification> classifications = stackalloc Classification[5];
		SelectingSpanProcessor.Process(Predictor.Output, classifications);
		return classifications.Length;
	}

	[Benchmark]
	public Classification[] Array()
	{
		ArrayProcessor.ClassificationsLimit = 5;
		return ArrayProcessor.Process(Predictor.Output);
	}

	[Benchmark]
	public Classification[] SelectingArray()
	{
		return SelectingArrayProcessor.Process(Predictor.Output);
	}

	private static readonly Predictor Predictor;
	private static readonly YoloClassificationProcessor SingleProcessor = new();
	private static readonly YoloClassificationsSpanProcessor SpanProcessor = new();
	private static readonly YoloClassificationsSelectingSpanProcessor SelectingSpanProcessor = new();
	private static readonly YoloClassificationsProcessor ArrayProcessor = new(new YoloClassificationsSpanProcessor());
	private static readonly YoloClassificationsProcessor SelectingArrayProcessor = new(new YoloClassificationsSelectingSpanProcessor());
}