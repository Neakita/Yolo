using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Benchmark;

public sealed class BenchmarkExecutionProvider
{
	public static BenchmarkExecutionProvider CPU => new("CPU", () => new SessionOptions());
	public static BenchmarkExecutionProvider Cuda => new("Cuda", () => SessionOptions.MakeSessionOptionWithCudaProvider());

	public string Designation { get; }

	public BenchmarkExecutionProvider(string designation, Func<SessionOptions> sessionOptionsFactory)
	{
		Designation = designation;
		_sessionOptionsFactory = sessionOptionsFactory;
	}

	public SessionOptions CreateSessionOptions()
	{
		return _sessionOptionsFactory();
	}

	public override string ToString()
	{
		return Designation;
	}

	private readonly Func<SessionOptions> _sessionOptionsFactory;
}