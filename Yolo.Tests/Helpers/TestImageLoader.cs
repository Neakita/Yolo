using CommunityToolkit.Diagnostics;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
		return imageData.AsMemory2D(image.Width, image.Height);
	}
}