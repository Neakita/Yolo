namespace TensorWeaver.Tests.Data;

public class ModelInfo
{
	public string Name { get; }
	public ushort Resolution { get; }

	public ModelInfo(string name, ushort resolution)
	{
		Name = name;
		Resolution = resolution;
	}
}