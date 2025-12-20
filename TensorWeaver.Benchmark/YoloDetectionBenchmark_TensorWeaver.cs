#pragma warning disable CS8618 // Nullability

using BenchmarkDotNet.Attributes;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.InputProcessing;
using TensorWeaver.OutputData;
using TensorWeaver.Yolo;

namespace TensorWeaver.Benchmark;

public partial class YoloDetectionBenchmark
{
	[GlobalSetup(Target = nameof(TensorWeaverPredict))]
	public void SetupTensorWeaver()
	{
		var options = ExecutionProvider.CreateSessionOptions();
		_tensorWeaverPredictor = new Predictor(Model.ReadData(), options);
		_imageDataForTensorWeaver = ImageInfo.GetPixels<Argb32>();
		var yoloMetadata = YoloMetadata.Parse(_tensorWeaverPredictor.Session);
		_outputProcessor = new YoloV8DetectionsSpanProcessor(yoloMetadata);
	}

	[GlobalCleanup(Target = nameof(TensorWeaverPredict))]
	public void CleanUpTensorWeaver()
	{
		_tensorWeaverPredictor.Dispose();
	}

	[Benchmark(Baseline = true)]
	public int TensorWeaverPredict()
	{
		_tensorWeaverPredictor.SetInput(_imageDataForTensorWeaver.Span, InputProcessor);
		_tensorWeaverPredictor.Predict();
		Span<Detection> detections = stackalloc Detection[20];
		return _outputProcessor.Process(_tensorWeaverPredictor.Output, detections);
	}

	private static readonly ResizingInputProcessor<Argb32> InputProcessor = ImageSharpInputProcessors.Argb32.WithResizing();

	private YoloV8DetectionsSpanProcessor _outputProcessor;
	private Predictor _tensorWeaverPredictor;
	private ReadOnlyMemory2D<Argb32> _imageDataForTensorWeaver;
}