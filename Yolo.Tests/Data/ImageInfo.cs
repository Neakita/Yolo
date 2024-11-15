namespace Yolo.Tests.Data;

public class ImageInfo
{
	public string Name { get; }
	public ushort Resolution { get; }

	public ImageInfo(string name, ushort resolution)
	{
		Name = name;
		Resolution = resolution;
	}
}