#nullable disable

using BenchmarkDotNet.Attributes;
using CommunityToolkit.HighPerformance;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace TensorWeaver.Benchmark;

public partial class NearestNeighborResizeBenchmark
{
	[GlobalSetup(Target = nameof(ImageSharpResize))]
	public void ImageSharpSetup()
	{
		var input = new Argb32[InputWidth * InputHeight];
		Random.NextBytes(input.AsSpan().AsBytes());
		_image = Image.LoadPixelData(input, InputWidth, InputHeight);
		_resizeOptions = new ResizeOptions
		{
			Size = new Size(OutputWidth, OutputHeight),
			Sampler = new NearestNeighborResampler()
		};
	}

	[IterationCleanup(Target = nameof(ImageSharpResize))]
	public void ImageSharpCleanUp()
	{
		_resized.Dispose();
		_resized = null;
	}

	[Benchmark]
	public void ImageSharpResize()
	{
		_resized = _image.Clone(context => context.Resize(_resizeOptions));
	}

	private Image _image;
	private Image _resized;
	private ResizeOptions _resizeOptions;
}