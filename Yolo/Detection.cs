namespace Yolo;

public readonly struct Detection
{
	public Classification Classification { get; }
	public Bounding Bounding { get; }
	public ushort ClassId => Classification.ClassId;
	public float Confidence => Classification.Confidence;

	public Detection(Classification classification, Bounding bounding)
	{
		Classification = classification;
		Bounding = bounding;
	}
}