using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Tests.Inference;

public sealed class TestExecutionProvider
{
	public static TestExecutionProvider CPU => new()
	{
		Designation = "cpu",
		Factory = () => new SessionOptions()
	};

	public static TestExecutionProvider Cuda => new()
	{
		Designation = "cuda",
		Factory = () => SessionOptions.MakeSessionOptionWithCudaProvider()
	};

	public required string Designation { get; init; }
	public required Func<SessionOptions> Factory { get; init; }
}