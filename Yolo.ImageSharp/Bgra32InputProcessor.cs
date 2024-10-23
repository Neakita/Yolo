using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.ImageSharp;

public class Bgra32InputProcessor : Packed32InputProcessor<Bgra32>
{
	public static Bgra32InputProcessor Instance { get; } = new();

	public Bgra32InputProcessor() : base(2, 1, 0)
	{
	}
}