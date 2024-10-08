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
		NewMethod(packedPixels, RedChannelMask, RedChannelShiftAmount, redChannelTarget);
		NewMethod(packedPixels, GreenChannelMask, GreenChannelShiftAmount, greenChannelTarget);
		NewMethod(packedPixels, BlueChannelMask, BlueChannelShiftAmount, blueChannelTarget);
	}

	private static void NewMethod(
		ReadOnlySpan<uint> packedPixels,
		uint channelMask,
		int channelShift,
		Span<float> redChannelTarget)
	{
		Span<uint> channelValues = stackalloc uint[packedPixels.Length];
		TensorPrimitives.BitwiseAnd(packedPixels, channelMask, channelValues);
		TensorPrimitives.ShiftRightArithmetic(channelValues, channelShift, channelValues);
		TensorPrimitives.ConvertChecked<uint, float>(channelValues, redChannelTarget);
		TensorPrimitives.Divide(redChannelTarget, 255.0f, redChannelTarget);
	}

	private const uint RedChannelMask = 0x00_00_FF_00;
	private const uint GreenChannelMask = 0x00_FF_00_00;
	private const uint BlueChannelMask = 0xFF_00_00_00;
	private const int RedChannelShiftAmount = sizeof(byte) * 8 * 1;
	private const int GreenChannelShiftAmount = sizeof(byte) * 8 * 2;
	private const int BlueChannelShiftAmount = sizeof(byte) * 8 * 3;
}