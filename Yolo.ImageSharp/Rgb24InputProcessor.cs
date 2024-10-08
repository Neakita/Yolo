using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.ImageSharp;

public sealed class Rgb24InputProcessor : InputProcessor<Rgb24>
{
	public static Rgb24InputProcessor Instance { get; } = new();

	protected override void GetNormalizedPixelValues(Rgb24 pixel, out float red, out float green, out float blue)
	{
		red = pixel.R / 255f;
		green = pixel.G / 255f;
		blue = pixel.B / 255f;
	}
}