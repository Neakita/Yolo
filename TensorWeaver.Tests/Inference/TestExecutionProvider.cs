using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Tests.Inference;

public sealed class TestExecutionProvider
{
	static TestExecutionProvider()
	{
		try
		{
			using var session = SessionOptions.MakeSessionOptionWithCudaProvider();
		}
		catch
		{
			IsCudaAvailable = false;
			return;
		}

		IsCudaAvailable = true;
	}

	public static TestExecutionProvider CPU => new()
	{
		Designation = "cpu",
		Factory = () => new SessionOptions()
	};

	public static TestExecutionProvider Cuda => new()
	{
		Designation = "cuda",
		IsAvailable = IsCudaAvailable,
		Factory = () => SessionOptions.MakeSessionOptionWithCudaProvider()
	};

	private static readonly bool IsCudaAvailable;

	public required string Designation { get; init; }
	public bool IsAvailable { get; init; } = true;
	public required Func<SessionOptions> Factory { get; init; }
}