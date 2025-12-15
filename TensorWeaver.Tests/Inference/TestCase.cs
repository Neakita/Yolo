namespace TensorWeaver.Tests.Inference;

public sealed class TestCase
{
	public required ModelInfo Model { get; init; }
	public required TestExecutionProvider ExecutionProvider { get; init; }
	public required ImageInfo Image { get; init; }
	public required PredictorOutputHandler OutputHandler { get; init; }

	public override string ToString()
	{
		return $"{Model.FileName} on {Image.FileName} using {ExecutionProvider.Designation}";
	}
}