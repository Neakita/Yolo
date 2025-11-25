namespace TensorWeaver.InputData;

public readonly ref struct RGBChanneledSpans
{
	public Span<float> RedChannel { get; }
	public Span<float> GreenChannel { get; }
	public Span<float> BlueChannel { get; }

	public RGBChanneledSpans this[Range range]
	{
		get
		{
			var redChannel = RedChannel[range];
			var greenChannel = GreenChannel[range];
			var blueChannel = BlueChannel[range];
			return new RGBChanneledSpans(redChannel, greenChannel, blueChannel);
		}
	}

	public RGBChanneledSpans(Span<float> redChannel, Span<float> greenChannel, Span<float> blueChannel)
	{
		RedChannel = redChannel;
		GreenChannel = greenChannel;
		BlueChannel = blueChannel;
	}
}