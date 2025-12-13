using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TensorWeaver.InputProcessing;

public sealed class ResizingInputProcessor<TPixel>(InputProcessor<TPixel> inner, Resizer resizer) : InputProcessor<TPixel>
{
	public void ProcessInput(ReadOnlySpan2D<TPixel> pixels, DenseTensor<float> tensor)
	{
		var width = tensor.Dimensions[3];
		var height = tensor.Dimensions[2];
		if (pixels.Width == width && pixels.Height == height)
			inner.ProcessInput(pixels, tensor);
		else
		{
			EnsureBufferCapacity(width * height);
			Span2D<TPixel> bufferSpan = new(_buffer, height, width);
			resizer.Resize(pixels, bufferSpan);
			inner.ProcessInput(bufferSpan, tensor);
		}
	}

	private TPixel[] _buffer = Array.Empty<TPixel>();

	private void EnsureBufferCapacity(int capacity)
	{
		if (_buffer.Length < capacity)
			_buffer = new TPixel[capacity];
	}
}