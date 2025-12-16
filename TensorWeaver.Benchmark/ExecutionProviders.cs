namespace TensorWeaver.Benchmark;

public static class ExecutionProviders
{
	public static IEnumerable<BenchmarkExecutionProvider> All =>
	[
		BenchmarkExecutionProvider.CPU,
		BenchmarkExecutionProvider.Cuda
	];
}