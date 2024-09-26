namespace Yolo;

public readonly struct Size
{
	public ushort Width { get; }
	public ushort Height { get; }

	public Size(ushort width, ushort height)
	{
		Width = width;
		Height = height;
	}
}