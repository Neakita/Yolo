namespace TensorWeaver.Tests.Data;

public class ModelInfo
{
	public required string Name { get; init; }
	public required ushort Resolution { get; init; }
	public string WebUrl { get; init; } = string.Empty;
}