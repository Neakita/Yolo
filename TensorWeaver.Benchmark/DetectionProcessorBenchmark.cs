using BenchmarkDotNet.Attributes;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.Benchmark.Models;
using TensorWeaver.ImageSharp;
using TensorWeaver.OutputData;
using TensorWeaver.Yolo;

namespace TensorWeaver.Benchmark;

public class DetectionProcessorBenchmark
{
	public DetectionProcessorBenchmark()
	{
		_predictor = new Predictor(DetectionModels.V8.First().ReadData(), new SessionOptions());
		var metadata = YoloMetadata.Parse(_predictor.Session);
		_listProcessor = new YoloV8DetectionsProcessor(metadata);
		_spanProcessor = new YoloV8DetectionsSpanProcessor(metadata);
		_predictor.SetInput(Images.Bus.GetPixels<Argb32>().Span, ImageSharpInputProcessors.Argb32.WithResizing());
	}

	[Benchmark]
	public List<Detection> List()
	{
		return _listProcessor.Process(_predictor.Output);
	}

	[Benchmark]
	public int Span()
	{
		Span<Detection> detections = stackalloc Detection[20];
		return _spanProcessor.Process(_predictor.Output, detections);
	}

	private readonly Predictor _predictor;
	private readonly YoloV8DetectionsProcessor _listProcessor;
	private readonly YoloV8DetectionsSpanProcessor _spanProcessor;
}