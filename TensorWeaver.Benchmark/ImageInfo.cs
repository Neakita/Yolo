using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace TensorWeaver.Benchmark;

public sealed class ImageInfo
{
	private const string ImagesDirectory = "Images";

	public string FileName { get; }

	public ImageInfo(string fileName)
	{
		FileName = fileName;
	}

	public Image Load()
	{
		return Image.Load(FilePath);
	}

	public Image<TPixel> Load<TPixel>() where TPixel : unmanaged, IPixel<TPixel>
	{
		return Image.Load<TPixel>(FilePath);
	}

	public ReadOnlyMemory2D<TPixel> GetPixels<TPixel>() where TPixel : unmanaged, IPixel<TPixel>
	{
		var image = Load<TPixel>();
		if (image.DangerousTryGetSinglePixelMemory(out var memory))
			return memory.ToArray().AsMemory().AsMemory2D(image.Height, image.Width);
		var pixels = new TPixel[image.Height, image.Width];
		for (int i = 0; i < image.Height; i++)
		{
			var sourceRow = image.DangerousGetPixelRowMemory(i);
			var targetRow = pixels.GetRowSpan(i);
			sourceRow.Span.CopyTo(targetRow);
		}
		return pixels;
	}

	private string FilePath => Path.Combine(ImagesDirectory, FileName);
}