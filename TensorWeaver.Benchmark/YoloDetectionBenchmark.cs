#pragma warning disable CS8618 // Nullability

using BenchmarkDotNet.Attributes;
using TensorWeaver.Benchmark.Models;

namespace TensorWeaver.Benchmark;

public partial class YoloDetectionBenchmark
{
	[ParamsSource(typeof(DetectionModels), nameof(DetectionModels.V8))]
	public ModelInfo Model;

	[ParamsSource(typeof(ExecutionProviders), nameof(ExecutionProviders.All))]
	public BenchmarkExecutionProvider ExecutionProvider;

	private static ImageInfo ImageInfo => new("bus.png");
}