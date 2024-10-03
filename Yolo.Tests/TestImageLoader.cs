using CommunityToolkit.Diagnostics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.Tests;

internal static class TestImageLoader
{
	public static ReadOnlyMemory2D<Rgb24> LoadImageData(string imageFileName)
	{
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var imageData));
		Vector2D<int> imageSize = new(image.Width, image.Height);
		return new ReadOnlyMemory2D<Rgb24>(imageSize, imageData);
	}
}