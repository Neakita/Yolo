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
		WriteNormalizedPixelValues(data.Span, redChannelData, greenChannelData, blueChannelData);
	}

	protected abstract void WriteNormalizedPixelValues(
		ReadOnlySpan<TPixel> pixels,
		Span<float> redChannelTarget,
		Span<float> greenChannelTarget,
		Span<float> blueChannelTarget);
}