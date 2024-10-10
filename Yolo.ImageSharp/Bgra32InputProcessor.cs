using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.ImageSharp;

public class Bgra32InputProcessor : ImageSharp32InputProcessor<Bgra32>
{
	public Bgra32InputProcessor() : base(2, 1, 0)
	{
	}
}