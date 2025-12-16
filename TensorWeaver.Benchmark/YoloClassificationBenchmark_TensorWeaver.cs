#pragma warning disable CS8618 // Nullability

using BenchmarkDotNet.Attributes;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.InputProcessing;
using TensorWeaver.OutputData;
using TensorWeaver.Yolo;

namespace TensorWeaver.Benchmark;

public partial class YoloClassificationBenchmark
{
	[GlobalSetup(Target = nameof(TensorWeaverPredict))]
	public void SetupTensorWeaver()
	{
		var options = ExecutionProvider.CreateSessionOptions();
		_tensorWeaverPredictor = new Predictor(Model.ReadData(), options);
		_imageDataForTensorWeaver = ImageInfo.GetPixels<Argb32>();
	}

	[GlobalCleanup(Target = nameof(TensorWeaverPredict))]
	public void CleanUpTensorWeaver()
	{
		_tensorWeaverPredictor.Dispose();
	}

	[Benchmark(Baseline = true)]
	public Classification TensorWeaverPredict()
	{
		_tensorWeaverPredictor.SetInput(_imageDataForTensorWeaver.Span, InputProcessor);
		_tensorWeaverPredictor.Predict();
		return OutputProcessor.Process(_tensorWeaverPredictor.Output);
	}

	private static readonly ResizingInputProcessor<Argb32> InputProcessor = ImageSharpInputProcessors.Argb32.WithResizing();
	private static readonly YoloClassificationProcessor OutputProcessor = new();

	private Predictor _tensorWeaverPredictor;
	private ReadOnlyMemory2D<Argb32> _imageDataForTensorWeaver;
}