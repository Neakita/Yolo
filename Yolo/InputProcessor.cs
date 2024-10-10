using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yolo;

public abstract class InputProcessor<TPixel>
	where TPixel : unmanaged
{
	public void ProcessInput(ReadOnlySpan2D<TPixel> data, DenseTensor<float> tensor)
	{
		var colorChannelStride = tensor.Strides[1];
		const int redChannelStart = 0;
		var greenChannelStart = colorChannelStride * 1;
		var blueChannelStart = colorChannelStride * 2;
		var targetSpan = tensor.Buffer.Span;
		Span<float> redChannelData = targetSpan.Slice(redChannelStart, colorChannelStride);
		Span<float> greenChannelData = targetSpan.Slice(greenChannelStart, colorChannelStride);
		Span<float> blueChannelData = targetSpan.Slice(blueChannelStart, colorChannelStride);
		RGBChanneledSpans spans = new(redChannelData, greenChannelData, blueChannelData);
		if (data.TryGetSpan(out var pixels))
		{
			WriteNormalizedPixelValues(pixels, spans);
			return;
		}
		for (ushort i = 0; i < data.Height; i++)
		{
			pixels = data.GetRowSpan(i);
			WriteNormalizedPixelValues(pixels, spans[(data.Width * i)..]);
		}
	}

	protected abstract void WriteNormalizedPixelValues(ReadOnlySpan<TPixel> pixels, RGBChanneledSpans target);
}