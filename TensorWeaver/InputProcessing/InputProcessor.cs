using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime.Tensors;
using TensorWeaver.InputData;

namespace TensorWeaver.InputProcessing;

public sealed class InputProcessor<TPixel> where TPixel : unmanaged
{
	public InputProcessor(byte redChannelPosition, byte greenChannelPosition, byte blueChannelPosition)
	{
		_redChannelMask = ComputeMask(redChannelPosition);
		_greenChannelMask = ComputeMask(greenChannelPosition);
		_blueChannelMask = ComputeMask(blueChannelPosition);
		_redChannelShift = ComputeShift(redChannelPosition);
		_greenChannelShift = ComputeShift(greenChannelPosition);
		_blueChannelShift = ComputeShift(blueChannelPosition);
	}

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

	private const uint ChannelMask = 0xFF;
	private readonly uint _redChannelMask;
	private readonly uint _greenChannelMask;
	private readonly uint _blueChannelMask;
	private readonly int _redChannelShift;
	private readonly int _greenChannelShift;
	private readonly int _blueChannelShift;

	private void WriteNormalizedPixelValues(
		ReadOnlySpan<TPixel> pixels,
		RGBChanneledSpans target)
	{
		var packedPixels = MemoryMarshal.Cast<TPixel, uint>(pixels);
		WriteChannelValues(packedPixels, _redChannelMask, _redChannelShift, target.RedChannel);
		WriteChannelValues(packedPixels, _greenChannelMask, _greenChannelShift, target.GreenChannel);
		WriteChannelValues(packedPixels, _blueChannelMask, _blueChannelShift, target.BlueChannel);
	}

	private static void WriteChannelValues(
		ReadOnlySpan<uint> packedPixels,
		uint channelMask,
		int channelShift,
		Span<float> channelTarget)
	{
		var bufferArray = ArrayPool<uint>.Shared.Rent(packedPixels.Length);
		var bufferSpan = bufferArray.AsSpan()[..packedPixels.Length];
		TensorPrimitives.BitwiseAnd(packedPixels, channelMask, bufferSpan);
		TensorPrimitives.ShiftRightArithmetic(bufferSpan, channelShift, bufferSpan);
		TensorPrimitives.ConvertChecked(bufferSpan, channelTarget);
		TensorPrimitives.Divide(channelTarget, 255.0f, channelTarget);
		ArrayPool<uint>.Shared.Return(bufferArray);
	}

	private static uint ComputeMask(byte redChannelPosition)
	{
		return ChannelMask << (8 * redChannelPosition);
	}

	private static int ComputeShift(byte redChannelPosition)
	{
		return sizeof(byte) * 8 * redChannelPosition;
	}
}