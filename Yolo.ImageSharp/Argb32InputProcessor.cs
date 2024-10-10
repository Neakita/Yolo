using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.ImageSharp;

public sealed class Argb32InputProcessor : ImageSharp32InputProcessor<Argb32>
{
	public static Argb32InputProcessor Instance { get; } = new();

	public Argb32InputProcessor() : base(1, 2, 3)
	{
		
	}
}