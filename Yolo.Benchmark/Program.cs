using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace Yolo.Benchmark;

internal static class Program
{
	private static void Main(string[] args)
	{
		BenchmarkRunner.Run<MatchingSizeClassificationBenchmark>(DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator), args);
	}
}