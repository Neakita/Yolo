using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Tests.Inference;

public sealed class InferenceTestSession
{
	public static InferenceTestSession CPU => new()
	{
		Designation = "cpu",
		Factory = () => new SessionOptions()
	};

	public static InferenceTestSession Cuda => new()
	{
		Designation = "cuda",
		Factory = () => SessionOptions.MakeSessionOptionWithCudaProvider()
	};

	public required string Designation { get; init; }
	public required Func<SessionOptions> Factory { get; init; }
}