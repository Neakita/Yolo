namespace Yolo.Tests.Data;

public sealed class ImageSetInfo
{
	public IReadOnlyCollection<ushort> Resolutions { get; }
	public IEnumerable<ImageInfo> Images => Resolutions.Select(ToImage);

	public ImageSetInfo(string namePattern, IReadOnlyCollection<ushort> resolutions)
	{
		_namePattern = namePattern;
		Resolutions = resolutions;
	}

	private readonly string _namePattern;

	private ImageInfo ToImage(ushort resolution)
	{
		return new ImageInfo(FormatName(resolution), resolution);
	}

	private string FormatName(ushort resolution)
	{
		return string.Format(_namePattern, resolution);
	}
}