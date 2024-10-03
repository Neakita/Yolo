using System.Runtime.CompilerServices;
using SixLabors.ImageSharp;

namespace Yolo.Tests;

internal static class ImageSaver
{
	public static void Save(Image image, string modelFileName, string originalImageFileName, bool gpuUsed, [CallerMemberName] string testName = "")
	{
		var newImageFileName = $"[{modelFileName}{(gpuUsed ? "-gpu" : string.Empty)}]{originalImageFileName}";
		var directory = Path.Combine("Images", "Plotted", testName);
		Directory.CreateDirectory(directory);
		var newImageFilePath = Path.Combine(directory, newImageFileName);
		image.Save(newImageFilePath);
	}
}