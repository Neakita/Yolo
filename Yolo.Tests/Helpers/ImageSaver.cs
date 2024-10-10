using SixLabors.ImageSharp;

namespace Yolo.Tests.Helpers;

internal static class ImageSaver
{
	public static void Save(Image image, string modelFileName, string originalImageFileName, bool gpuUsed, string testName)
	{
		var newImageFileName = $"[{modelFileName}{(gpuUsed ? "-gpu" : string.Empty)}]{originalImageFileName}";
		var directory = Path.Combine("Images", "Plotted", testName);
		Directory.CreateDirectory(directory);
		var newImageFilePath = Path.Combine(directory, newImageFileName);
		image.Save(newImageFilePath);
	}
}