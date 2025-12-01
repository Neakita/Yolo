using System.Buffers;
using System.Numerics.Tensors;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TensorWeaver.InputProcessing;

public sealed class InputProcessor<TPixel> where TPixel : unmanaged
{
	public InputProcessor(params IReadOnlyCollection<byte> channelPositions)
	{
		_channelMasks = channelPositions.Select(ComputeMask).ToArray();
		_channelShifts = channelPositions.Select(ComputeShift).ToArray();
	}

	public void ProcessInput(ReadOnlySpan2D<TPixel> pixels, DenseTensor<float> tensor)
	{
		var channelsCount = tensor.Dimensions[1];
		var channelsStride = tensor.Strides[1];
		var targetSpan = tensor.Buffer.Span;
		for (int i = 0; i < channelsCount; i++)
		{
			var channelStart = channelsStride * i;
			var targetChannelSpan = targetSpan.Slice(channelStart, channelsStride);
			WriteChannelValues(pixels, i, targetChannelSpan);
		}
	}

	private const uint ChannelMask = 0xFF;
	private readonly uint[] _channelMasks;
	private readonly int[] _channelShifts;

	private static uint ComputeMask(byte channelPosition)
	{
		return ChannelMask << (8 * channelPosition);
	}

	private static int ComputeShift(byte channelPosition)
	{
		return 8 * channelPosition;
	}

	private void WriteChannelValues(ReadOnlySpan2D<TPixel> pixels, int channelIndex, Span<float> targetChannelSpan)
	{
		if (pixels.TryGetSpan(out var pixelsSpan))
		{
			WriteChannelValues(pixelsSpan, channelIndex, targetChannelSpan);
			return;
		}
		for (int i = 0; i < pixels.Height; i++)
		{
			var pixelsRowSpan = pixels.GetRowSpan(i);
			WriteChannelValues(pixelsRowSpan, channelIndex, targetChannelSpan[(pixels.Width * i)..]);
		}
	}

	private void WriteChannelValues(
		ReadOnlySpan<TPixel> pixels,
		int channelIndex,
		Span<float> channelTarget)
	{
		var packedPixels = pixels.Cast<TPixel, uint>();
		WriteChannelValues(packedPixels, channelIndex, channelTarget);
	}

	private void WriteChannelValues(
		ReadOnlySpan<uint> source,
		int channelIndex,
		Span<float> target)
	{
		var bufferArray = ArrayPool<uint>.Shared.Rent(source.Length);
		var buffer = bufferArray.AsSpan()[..source.Length];
		var mask = _channelMasks[channelIndex];
		var shift = _channelShifts[channelIndex]; 
		TensorPrimitives.BitwiseAnd(source, mask, buffer);
		TensorPrimitives.ShiftRightArithmetic(buffer, shift, buffer);
		TensorPrimitives.ConvertChecked(buffer, target);
		const float byteMaxValue = byte.MaxValue;
		TensorPrimitives.Divide(target, byteMaxValue, target);
		ArrayPool<uint>.Shared.Return(bufferArray);
	}
}