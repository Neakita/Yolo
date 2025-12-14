using BenchmarkDotNet.Attributes;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.InputProcessing;

namespace TensorWeaver.Benchmark;

public partial class NearestNeighborResizeBenchmark
{
	[GlobalSetup(Target = nameof(OnSpanResize))]
	public void OnSpanSetup()
	{
		var input = new Argb32[InputWidth * InputHeight];
		_nearestNeighborInput = new Memory2D<Argb32>(input, InputHeight, InputWidth);
		Random.NextBytes(input.AsSpan().AsBytes());
		_nearestNeighborOutput = new Memory2D<Argb32>(new Argb32[OutputWidth * OutputHeight], OutputHeight, OutputWidth);
	}

	[IterationCleanup(Target = nameof(OnSpanResize))]
	public void EraseSpanOutput()
	{
		_nearestNeighborOutput.Span.Fill(default);
	}

	[Benchmark(Baseline = true)]
	public void OnSpanResize()
	{
		NearestNeighborResizer.Resize(_nearestNeighborInput.Span, _nearestNeighborOutput.Span);
	}

	private static readonly NearestNeighbourImageResizer NearestNeighborResizer = new();
	private ReadOnlyMemory2D<Argb32> _nearestNeighborInput;
	private Memory2D<Argb32> _nearestNeighborOutput;
}