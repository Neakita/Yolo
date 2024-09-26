namespace Yolo;

public sealed class Detection : Prediction
{
	public Detection(byte classId, float confidence) : base(classId, confidence)
	{
	}
}