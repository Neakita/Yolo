using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yolo;

public abstract class InputProcessor<TPixel>
	where TPixel : unmanaged
{
	public void ProcessInput(ReadOnlySpan<TPixel> data, DenseTensor<float> tensor)
	{
		var redChannelStart = tensor.Strides[1] * 0;
		var greenChannelStart = tensor.Strides[1] * 1;
		var blueChannelStart = tensor.Strides[1] * 2;
		var height = tensor.Strides[2];
		var tensorSpan = tensor.Buffer.Span;
		var width = tensor.Strides[1] / height;
		for (ushort y = 0; y < width; y++)
		for (ushort x = 0; x < height; x++)
		{
			var index = x + width * y;
			var pixel = data[index];
			WritePixel(tensorSpan, index, pixel, redChannelStart, greenChannelStart, blueChannelStart);
		}
	}

	protected abstract void GetNormalizedPixelValues(TPixel pixel, out float red, out float green, out float blue);

	private void WritePixel(Span<float> target, int index, TPixel pixel, int redChannelStart, int greenChannelStart, int blueChannelStart)
	{
		GetNormalizedPixelValues(pixel, out float red, out float green, out float blue);
		target[index] = red;
		target[index + greenChannelStart - redChannelStart] = green;
		target[index + blueChannelStart - redChannelStart] = blue;
	}
}