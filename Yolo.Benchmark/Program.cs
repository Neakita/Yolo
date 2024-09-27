using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace Yolo.Benchmark;

internal static class Program
{
	private static void Main(string[] args)
	{
		BenchmarkRunner.Run(Assembly.GetAssembly(typeof(Program))!, DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator), args);
	}
}