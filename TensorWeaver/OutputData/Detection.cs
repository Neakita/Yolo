namespace TensorWeaver.OutputData;

public readonly struct Detection
{
	public Classification Classification { get; }
	public Bounding Bounding { get; }
	public byte ClassId => Classification.ClassId;
	public float Confidence => Classification.Confidence;

	public ushort Index { get; }

	public Detection(Classification classification, Bounding bounding, ushort index)
	{
		Classification = classification;
		Bounding = bounding;
		Index = index;
	}
}