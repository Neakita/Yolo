using BenchmarkDotNet.Attributes;

namespace TensorWeaver.Benchmark;

public partial class NearestNeighborResizeBenchmark
{
	[Params(320, 1920)] public int InputWidth { get; set; }
	[Params(320, 1920)] public int InputHeight { get; set; }
	[Params(320, 1920)] public int OutputWidth { get; set; }
	[Params(320, 1920)] public int OutputHeight { get; set; }

	private static Random Random => new(0);
}