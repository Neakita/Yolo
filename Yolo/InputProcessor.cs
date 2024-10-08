using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yolo;

public abstract class InputProcessor<TPixel>
	where TPixel : unmanaged
{
	public void ProcessInput(ReadOnlySpan2D<TPixel> data, DenseTensor<float> tensor)
	{
		var colorChannelStride = tensor.Strides[1];
		var greenChannelStart = colorChannelStride * 1;
		var blueChannelStart = colorChannelStride * 2;
		var tensorSpan = tensor.Buffer.Span;
		var width = tensor.Strides[2];
		var height = colorChannelStride / width;
		for (ushort y = 0; y < height; y++)
		{
			var sourceRow = data[y];
			var rowStart = width * y;
			for (ushort x = 0; x < width; x++)
			{
				var pixel = sourceRow[x];
				var index = rowStart + x;
				WritePixel(tensorSpan, index, pixel, greenChannelStart, blueChannelStart);
			}
		}
	}

	protected abstract void GetNormalizedPixelValues(TPixel pixel, out float red, out float green, out float blue);

	private void WritePixel(Span<float> target, int index, TPixel pixel, int greenChannelStart, int blueChannelStart)
	{
		GetNormalizedPixelValues(pixel,
			out target[index],
			out target[greenChannelStart + index],
			out target[blueChannelStart + index]);
	}
}