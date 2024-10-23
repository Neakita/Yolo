using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Yolo.InputData;

namespace Yolo.InputProcessing;

public class Packed32InputProcessor<TPixel> : InputProcessor<TPixel> where TPixel : unmanaged
{
	protected Packed32InputProcessor(byte redChannelPosition, byte greenChannelPosition, byte blueChannelPosition)
	{
		_redChannelMask = ComputeMask(redChannelPosition);
		_greenChannelMask = ComputeMask(greenChannelPosition);
		_blueChannelMask = ComputeMask(blueChannelPosition);
		_redChannelShift = ComputeShift(redChannelPosition);
		_greenChannelShift = ComputeShift(greenChannelPosition);
		_blueChannelShift = ComputeShift(blueChannelPosition);
	}

	protected override void WriteNormalizedPixelValues(
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
		TensorPrimitives.ConvertChecked<uint, float>(bufferSpan, channelTarget);
		TensorPrimitives.Divide(channelTarget, 255.0f, channelTarget);
		ArrayPool<uint>.Shared.Return(bufferArray);
	}

	private const uint ChannelMask = 0xFF;
	private readonly uint _redChannelMask;
	private readonly uint _greenChannelMask;
	private readonly uint _blueChannelMask;
	private readonly int _redChannelShift;
	private readonly int _greenChannelShift;
	private readonly int _blueChannelShift;

	private static uint ComputeMask(byte redChannelPosition)
	{
		return ChannelMask << (8 * redChannelPosition);
	}

	private static int ComputeShift(byte redChannelPosition)
	{
		return sizeof(byte) * 8 * redChannelPosition;
	}
}