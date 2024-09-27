namespace Yolo;

public sealed class Detection : Prediction
{
	public Bounding Bounding { get; }

	public Detection(ushort classId, float confidence, Bounding bounding) : base(classId, confidence)
	{
		Bounding = bounding;
	}
}