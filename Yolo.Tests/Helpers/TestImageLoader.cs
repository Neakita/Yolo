using CommunityToolkit.Diagnostics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Yolo.Tests.Data;

namespace Yolo.Tests.Helpers;

internal static class TestImageLoader
{
	public static Image<Argb32> LoadImage(string imageFileName)
	{
		var imageFilePath = Path.Combine("Images", imageFileName);
		return Image.Load<Argb32>(imageFilePath);
	}

	public static ReadOnlyMemory2D<Argb32> ExtractImageData(Image<Argb32> image)
	{
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var imageData));
		Vector2D<int> imageSize = new(image.Width, image.Height);
		return new ReadOnlyMemory2D<Argb32>(imageSize, imageData);
	}
}