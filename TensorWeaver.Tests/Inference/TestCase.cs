namespace TensorWeaver.Tests.Inference;

public sealed class TestCase
{
	public required ModelInfo Model { get; init; }
	public required InferenceTestSession Session { get; init; }
	public required ImageInfo ImageInfo { get; init; }
	public required PredictorOutputHandler OutputHandler { get; init; }

	public override string ToString()
	{
		return $"{Model.FileName} on {ImageInfo.FileName} using {Session.Designation}";
	}
}