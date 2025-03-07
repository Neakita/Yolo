using SixLabors.ImageSharp.PixelFormats;
using Yolo.InputProcessing;

namespace Yolo.ImageSharp;

public sealed class Argb32InputProcessor : Packed32InputProcessor<Argb32>
{
	public static Argb32InputProcessor Instance { get; } = new();

	public Argb32InputProcessor() : base(1, 2, 3)
	{
		
	}
}