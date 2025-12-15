using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Tests.Inference;

public sealed class TestCase
{
	public required ModelInfo Model { get; init; }
	public required Func<SessionOptions> SessionOptionsFactory { get; init; }
	public required ImageInfo ImageInfo { get; init; }
	public required PredictorOutputHandler OutputHandler { get; init; }
}