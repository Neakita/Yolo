#pragma warning disable CS8618 // Nullability

using BenchmarkDotNet.Attributes;
using TensorWeaver.Benchmark.Models;

namespace TensorWeaver.Benchmark;

public partial class YoloClassificationBenchmark
{
	[ParamsSource(typeof(ClassificationModels), nameof(ClassificationModels.All))]
	public ModelInfo Model;

	[ParamsSource(typeof(ExecutionProviders), nameof(ExecutionProviders.All))]
	public BenchmarkExecutionProvider ExecutionProvider;

	private static ImageInfo ImageInfo => Images.Bus;
}