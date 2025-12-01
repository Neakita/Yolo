using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.InputProcessing;

namespace TensorWeaver.ImageSharp;

public static class ImageSharpInputProcessors
{
	public static InputProcessor<Argb32> Argb32 => new(1, 2, 3);
	public static InputProcessor<Bgra32> Bgra32 => new(2, 1, 0);
}