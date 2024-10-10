using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp.PixelFormats;

namespace Yolo.ImageSharp;

public sealed class Argb32InputProcessor : InputProcessor<Argb32>
{
	public static Argb32InputProcessor Instance { get; } = new();

	protected override void WriteNormalizedPixelValues(
		ReadOnlySpan<Argb32> pixels,
		Span<float> redChannelTarget,
		Span<float> greenChannelTarget,
		Span<float> blueChannelTarget)
	{
		ReadOnlySpan<uint> packedPixels = MemoryMarshal.Cast<Argb32, uint>(pixels);
		WriteChannelValues(packedPixels, RedChannelMask, RedChannelShiftAmount, redChannelTarget);
		WriteChannelValues(packedPixels, GreenChannelMask, GreenChannelShiftAmount, greenChannelTarget);
		WriteChannelValues(packedPixels, BlueChannelMask, BlueChannelShiftAmount, blueChannelTarget);
	}

	private static void WriteChannelValues(
		ReadOnlySpan<uint> packedPixels,
		uint channelMask,
		int channelShift,
		Span<float> redChannelTarget)
	{
		var bufferArray = ArrayPool<uint>.Shared.Rent(packedPixels.Length);
		var bufferSpan = bufferArray.AsSpan()[..packedPixels.Length];
		TensorPrimitives.BitwiseAnd(packedPixels, channelMask, bufferSpan);
		TensorPrimitives.ShiftRightArithmetic(bufferSpan, channelShift, bufferSpan);
		TensorPrimitives.ConvertChecked<uint, float>(bufferSpan, redChannelTarget);
		TensorPrimitives.Divide(redChannelTarget, 255.0f, redChannelTarget);
		ArrayPool<uint>.Shared.Return(bufferArray);
	}

	private const uint RedChannelMask = 0x00_00_FF_00;
	private const uint GreenChannelMask = 0x00_FF_00_00;
	private const uint BlueChannelMask = 0xFF_00_00_00;
	private const int RedChannelShiftAmount = sizeof(byte) * 8 * 1;
	private const int GreenChannelShiftAmount = sizeof(byte) * 8 * 2;
	private const int BlueChannelShiftAmount = sizeof(byte) * 8 * 3;
}