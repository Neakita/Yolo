namespace Yolo.OutputData;

public readonly struct Detection
{
	public Classification Classification { get; }
	public Bounding Bounding { get; }
	public ushort ClassId => Classification.ClassId;
	public float Confidence => Classification.Confidence;

	internal ushort Index { get; }

	internal Detection(Classification classification, Bounding bounding, ushort index)
	{
		Classification = classification;
		Bounding = bounding;
		Index = index;
	}
}