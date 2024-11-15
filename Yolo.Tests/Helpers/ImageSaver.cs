using System.Text;
using SixLabors.ImageSharp;

namespace Yolo.Tests.Helpers;

internal static class ImageSaver
{
	public static void Save(Image image, string modelFileName, string originalImageFileName, bool gpuUsed, string testName)
	{
		StringBuilder fileNameBuilder = new(3);
		fileNameBuilder.Append(modelFileName);
		if (gpuUsed)
			fileNameBuilder.Append("-gpu");
		fileNameBuilder.Append(originalImageFileName);
		var fileName = fileNameBuilder.ToString();
		var directory = Path.Combine("Images", "Plotted", testName);
		Directory.CreateDirectory(directory);
		var newImageFilePath = Path.Combine(directory, fileName);
		image.Save(newImageFilePath);
	}
}