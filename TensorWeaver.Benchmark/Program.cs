using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Diagnostics.dotTrace;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;

namespace TensorWeaver.Benchmark;

internal static class Program
{
	private static void Main(string[] args)
	{
		var options = DefaultConfig.Instance
			.WithOptions(ConfigOptions.DisableOptimizationsValidator)
			.WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(100))
			.AddDiagnoser(new DotTraceDiagnoser())
			.AddDiagnoser(MemoryDiagnoser.Default);
		BenchmarkRunner.Run(typeof(Program).Assembly, options, args);
	}
}